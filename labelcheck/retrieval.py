"""Поиск по корпусу регламентов: BM25, векторный (Qdrant), гибрид RRF.

Три метода с одинаковым интерфейсом — каждый принимает текст запроса и
возвращает список (chunk_id, score) от лучшего к худшему:

- bm25_search:   точное совпадение слов (Е-коды, числа, термины);
- vector_search: смысловая близость (парафразы: «калорийность» найдёт
                 «энергетическую ценность»);
- hybrid_search: слияние двух выдач по местам (Reciprocal Rank Fusion) —
                 скоры BM25 и косинусная близость живут в несравнимых
                 шкалах, поэтому складываются не скоры, а ранги.

Все параметры — в labelcheck/config.yaml. Векторы берутся из кэша
data/embeddings.npz (создаёт ingestion/index.py); BM25 строится в памяти
при старте за ~2 секунды, хранить его на диске незачем.
"""

import json
import re
from pathlib import Path

import numpy as np
import snowballstemmer
import yaml
from qdrant_client import QdrantClient
from qdrant_client import models as qm
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).parent.parent
CONFIG_PATH = Path(__file__).parent / "config.yaml"
CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"

# Токен — последовательность русских/латинских букв или цифр.
TOKEN_RE = re.compile(r"[а-яёa-z0-9]+")
# Е-код латиницей: отдельная буква e/E перед цифрами (уже после lowercase).
LATIN_E_CODE_RE = re.compile(r"^e(\d+)$")


def load_config(path: Path = CONFIG_PATH) -> dict:
    return yaml.safe_load(open(path, encoding="utf-8"))


def chunk_header(chunk: dict, fields: list[str]) -> str:
    """Шапка чанка из перечисленных в конфиге полей метаданных."""
    parts = []
    for f in fields:
        value = chunk.get(f)
        if value:
            parts.append(f"Приложение {value}" if f == "appendix" else str(value))
    return ". ".join(parts)


def chunk_to_text(chunk: dict, fields: list[str]) -> str:
    """Текст чанка с шапкой — одинаковый вход для BM25 и эмбеддингов."""
    header = chunk_header(chunk, fields)
    return f"{header}\n{chunk['text']}" if header else chunk["text"]


class Tokenizer:
    """Токенизация для BM25: lowercase → токены → нормализация Е-кодов → стемминг."""

    def __init__(self, bm25_cfg: dict):
        self.normalize_e = bm25_cfg.get("normalize_latin_e", False)
        lang = bm25_cfg.get("stemmer")
        self.stemmer = snowballstemmer.stemmer(lang) if lang else None

    def __call__(self, text: str) -> list[str]:
        tokens = TOKEN_RE.findall(text.lower())
        if self.normalize_e:
            tokens = [LATIN_E_CODE_RE.sub(r"е\1", t) for t in tokens]  # е — кириллица
        if self.stemmer:
            tokens = self.stemmer.stemWords(tokens)
        return tokens


def rrf_fuse(rankings: list[list[str]], rrf_k: int, top_k: int) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: балл документа = сумма 1/(rrf_k + место)
    по всем выдачам, где он встретился. Скоры методов не используются —
    только позиции, поэтому калибровка шкал не нужна."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for pos, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + pos)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:top_k]


class Retriever:
    """Держит оба индекса и отвечает на запросы тремя методами.

    openai_client передаётся снаружи (нужен только векторным методам);
    BM25 работает и без него — удобно для тестов без API.
    """

    def __init__(self, config: dict | None = None, openai_client=None):
        self.cfg = config or load_config()
        self.client = openai_client
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]
        self.chunk_ids = [c["chunk_id"] for c in self.chunks]
        self.regulation_of = {c["chunk_id"]: c["regulation_id"] for c in self.chunks}

        fields = self.cfg["embedding"]["header_fields"]
        self.tokenizer = Tokenizer(self.cfg["bm25"])
        texts = [chunk_to_text(c, fields) for c in self.chunks]
        self.bm25 = BM25Okapi([self.tokenizer(t) for t in texts])
        self._qdrant = None  # создаётся при первом векторном запросе

    # --- BM25 ---

    def bm25_search(self, query: str, k: int | None = None,
                    regulation: str | None = None) -> list[tuple[str, float]]:
        k = k or self.cfg["search"]["candidates"]
        scores = self.bm25.get_scores(self.tokenizer(query))
        order = np.argsort(scores)[::-1]  # индексы от лучшего к худшему
        result = []
        for i in order:
            cid = self.chunk_ids[i]
            if regulation and self.regulation_of[cid] != regulation:
                continue
            if scores[i] <= 0:
                break
            result.append((cid, float(scores[i])))
            if len(result) == k:
                break
        return result

    # --- Векторный поиск ---

    def _ensure_qdrant(self):
        """Ленивая инициализация: Qdrant собирается в памяти из npz-кэша
        при первом векторном запросе (mode: memory) или подключается
        к серверу (mode: server, docker-compose)."""
        if self._qdrant is not None:
            return
        qcfg = self.cfg["qdrant"]
        if qcfg["mode"] == "server":
            self._qdrant = QdrantClient(url=qcfg["url"])
            return

        cache = ROOT / self.cfg["embedding"]["cache"]
        if not cache.exists():
            raise FileNotFoundError(
                f"нет кэша векторов {cache} — сначала: python ingestion/index.py")
        data = np.load(cache, allow_pickle=False)
        cached_ids = [i for i in data["chunk_ids"]]
        if list(cached_ids) != self.chunk_ids:
            raise ValueError("кэш векторов не совпадает с корпусом по составу "
                             "чанков — пересоздай: python ingestion/index.py --force")

        client = QdrantClient(":memory:")
        client.create_collection(
            collection_name=qcfg["collection"],
            vectors_config=qm.VectorParams(
                size=self.cfg["embedding"]["dimensions"],
                distance=qm.Distance.COSINE))
        client.upsert(
            collection_name=qcfg["collection"],
            points=[qm.PointStruct(
                id=i,
                vector=data["vectors"][i].tolist(),
                payload={"chunk_id": cid,
                         "regulation_id": self.regulation_of[cid]})
                for i, cid in enumerate(self.chunk_ids)])
        self._qdrant = client

    def embed_query(self, query: str) -> list[float]:
        if self.client is None:
            raise RuntimeError("для векторного поиска нужен openai_client")
        resp = self.client.embeddings.create(
            model=self.cfg["embedding"]["model"], input=[query])
        return resp.data[0].embedding

    def vector_search(self, query: str, k: int | None = None,
                      regulation: str | None = None) -> list[tuple[str, float]]:
        k = k or self.cfg["search"]["candidates"]
        self._ensure_qdrant()
        flt = None
        if regulation:
            flt = qm.Filter(must=[qm.FieldCondition(
                key="regulation_id", match=qm.MatchValue(value=regulation))])
        hits = self._qdrant.query_points(
            collection_name=self.cfg["qdrant"]["collection"],
            query=self.embed_query(query), limit=k, query_filter=flt).points
        return [(h.payload["chunk_id"], float(h.score)) for h in hits]

    # --- Гибрид ---

    def hybrid_search(self, query: str, k: int | None = None,
                      regulation: str | None = None) -> list[tuple[str, float]]:
        k = k or self.cfg["search"]["top_k"]
        n = self.cfg["search"]["candidates"]
        bm25_ids = [cid for cid, _ in self.bm25_search(query, n, regulation)]
        vec_ids = [cid for cid, _ in self.vector_search(query, n, regulation)]
        return rrf_fuse([bm25_ids, vec_ids], self.cfg["search"]["rrf_k"], k)
