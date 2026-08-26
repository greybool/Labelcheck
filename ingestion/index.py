"""Построение векторного индекса: чанки → эмбеддинги OpenAI → кэш npz.

Запуск из корня репозитория:  python ingestion/index.py [--force]

Результат — data/embeddings.npz: матрица векторов + chunk_id + SHA256-пломба
корпуса + имя модели. Поиск (labelcheck/retrieval.py) поднимает из этого
кэша Qdrant в памяти за секунды; при mode: server (docker-compose, День 10)
скрипт дополнительно заливает векторы на сервер.

Если кэш уже существует и совпадает с корпусом по пломбе и модели — API
не вызывается (повторный запуск бесплатен). --force пересоздаёт кэш.
Фактический расход токенов печатается по данным API, не по оценке.
"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.matcher import sha256_of  # та же пломба, что у ground truth
from labelcheck.retrieval import CHUNKS_PATH, chunk_to_text, load_config


def cache_is_fresh(cache: Path, corpus_sha: str, model: str) -> bool:
    """Кэш актуален: файл есть, пломба корпуса и модель совпадают."""
    if not cache.exists():
        return False
    data = np.load(cache, allow_pickle=False)
    return (str(data["corpus_sha256"]) == corpus_sha
            and str(data["model"]) == model)


def main() -> int:
    load_dotenv(ROOT / ".env")
    cfg = load_config()
    ecfg = cfg["embedding"]
    cache = ROOT / ecfg["cache"]
    corpus_sha = sha256_of(CHUNKS_PATH)

    if "--force" not in sys.argv and cache_is_fresh(cache, corpus_sha, ecfg["model"]):
        print(f"Кэш {cache.name} актуален (пломба корпуса совпадает) — "
              "эмбеддинги не пересчитываются. Пересоздать: --force")
        return 0

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    texts = [chunk_to_text(c, ecfg["header_fields"]) for c in chunks]
    print(f"Эмбеддинг {len(texts)} чанков, модель {ecfg['model']}, "
          f"батч {ecfg['batch_size']}…")

    client = OpenAI()  # ключ берётся из окружения (.env)
    vectors, total_tokens = [], 0
    for start in range(0, len(texts), ecfg["batch_size"]):
        batch = texts[start:start + ecfg["batch_size"]]
        resp = client.embeddings.create(model=ecfg["model"], input=batch)
        # API возвращает эмбеддинги в порядке входа; сортировка по index —
        # страховка от нарушения порядка.
        for item in sorted(resp.data, key=lambda d: d.index):
            vectors.append(item.embedding)
        total_tokens += resp.usage.total_tokens
        done = min(start + ecfg["batch_size"], len(texts))
        print(f"\r  {done}/{len(texts)} чанков, {total_tokens:,} токенов",
              end="", flush=True)
    print()

    matrix = np.asarray(vectors, dtype=np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        vectors=matrix,
        chunk_ids=np.asarray([c["chunk_id"] for c in chunks]),
        corpus_sha256=corpus_sha,
        model=ecfg["model"])
    cost = total_tokens / 1e6 * 0.02  # тариф text-embedding-3-small, $/1M токенов
    print(f"Готово: {matrix.shape[0]} векторов x{matrix.shape[1]} → {cache}")
    print(f"Фактический расход: {total_tokens:,} токенов ≈ ${cost:.3f}")

    if cfg["qdrant"]["mode"] == "server":
        from qdrant_client import QdrantClient
        from qdrant_client import models as qm
        qcfg = cfg["qdrant"]
        qc = QdrantClient(url=qcfg["url"])
        qc.recreate_collection(
            collection_name=qcfg["collection"],
            vectors_config=qm.VectorParams(size=matrix.shape[1],
                                           distance=qm.Distance.COSINE))
        qc.upsert(collection_name=qcfg["collection"],
                  points=[qm.PointStruct(id=i, vector=matrix[i].tolist(),
                                         payload={"chunk_id": c["chunk_id"],
                                                  "regulation_id": c["regulation_id"]})
                          for i, c in enumerate(chunks)])
        print(f"Векторы залиты на сервер Qdrant: {qcfg['url']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
