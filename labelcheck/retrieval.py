"""Поиск по корпусу регламентов: BM25, векторный (Qdrant), гибрид RRF.

Три метода с одинаковым интерфейсом — каждый принимает текст запроса и
возвращает список (chunk_id, score) от лучшего к худшему. Фильтр
regulation принимает один регламент строкой или список (у аспектов
чек-листа регламентов 1–4, см. labelcheck/aspects.yaml):

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


def qdrant_settings(cfg: dict) -> tuple[str, str]:
    """(mode, url): из config.yaml, с переопределением переменными окружения
    QDRANT_MODE и QDRANT_URL — так docker-compose переключает приложение на
    сервер Qdrant, а локальный запуск без докера остаётся в памяти."""
    import os
    qcfg = cfg["qdrant"]
    return (os.environ.get("QDRANT_MODE") or qcfg["mode"],
            os.environ.get("QDRANT_URL") or qcfg["url"])


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


def as_regulation_list(regulation: str | list[str] | None) -> list[str] | None:
    """Нормализация фильтра: строка → список из одного элемента."""
    if regulation is None:
        return None
    return [regulation] if isinstance(regulation, str) else list(regulation)


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
                    regulation: str | list[str] | None = None) -> list[tuple[str, float]]:
        k = k or self.cfg["search"]["candidates"]
        allowed = as_regulation_list(regulation)
        scores = self.bm25.get_scores(self.tokenizer(query))
        order = np.argsort(scores)[::-1]  # индексы от лучшего к худшему
        result = []
        for i in order:
            cid = self.chunk_ids[i]
            if allowed is not None and self.regulation_of[cid] not in allowed:
                continue
            if scores[i] <= 0:
                break
            result.append((cid, float(scores[i])))
            if len(result) == k:
                break
        return result

    # --- Векторный поиск ---

    def _ensure_qdrant(self):
        """Ленивая инициализация при первом векторном запросе.

        mode: memory — Qdrant собирается в памяти из npz-кэша (~3 с).
        mode: server — подключение к серверу (docker-compose, День 10); если
        коллекции на сервере нет или в ней не тот набор точек, она
        заливается из того же npz-кэша — источник истины векторов один,
        отдельной базы на сервере не ведём. Режим и адрес можно переопределить
        переменными окружения QDRANT_MODE / QDRANT_URL (compose ставит их,
        не трогая config.yaml)."""
        if self._qdrant is not None:
            return
        mode, url = qdrant_settings(self.cfg)
        name = self.cfg["qdrant"]["collection"]
        if mode == "server":
            client = QdrantClient(url=url)
            # Сервер в docker-compose поднимается параллельно с приложением:
            # первые секунды соединение может не приниматься — ждём, а не
            # падаем (до ~30 с).
            import time
            for attempt in range(10):
                try:
                    exists = client.collection_exists(name)
                    break
                except Exception:  # noqa: BLE001 — любая сетевая ошибка клиента
                    if attempt == 9:
                        raise
                    time.sleep(3)
            if exists and client.count(name, exact=True).count == len(self.chunk_ids):
                self._qdrant = client
                return
        elif mode == "memory":
            client = QdrantClient(":memory:")
        else:
            raise ValueError(f"qdrant.mode: ожидаю memory или server, получено {mode!r}")
        self._fill_collection(client, name)
        self._qdrant = client

    def _load_vector_cache(self):
        cache = ROOT / self.cfg["embedding"]["cache"]
        if not cache.exists():
            raise FileNotFoundError(
                f"нет кэша векторов {cache} — сначала: python ingestion/index.py")
        data = np.load(cache, allow_pickle=False)
        cached_ids = [i for i in data["chunk_ids"]]
        if list(cached_ids) != self.chunk_ids:
            raise ValueError("кэш векторов не совпадает с корпусом по составу "
                             "чанков — пересоздай: python ingestion/index.py --force")
        return data["vectors"]

    def _fill_collection(self, client, name: str):
        """(Пере)создать коллекцию и залить векторы из npz-кэша."""
        vectors = self._load_vector_cache()
        if client.collection_exists(name):
            client.delete_collection(name)
        client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(
                size=self.cfg["embedding"]["dimensions"],
                distance=qm.Distance.COSINE))
        batch = 256  # серверу — порциями, а не 2417 точек одним запросом
        for start in range(0, len(self.chunk_ids), batch):
            client.upsert(
                collection_name=name,
                points=[qm.PointStruct(
                    id=i,
                    vector=vectors[i].tolist(),
                    payload={"chunk_id": cid,
                             "regulation_id": self.regulation_of[cid]})
                    for i, cid in list(enumerate(self.chunk_ids))[start:start + batch]])

    def warm_up_vectors(self):
        """Собрать/залить коллекцию заранее (index.py в режиме server,
        прогрев при старте контейнера)."""
        self._ensure_qdrant()

    def embed_query(self, query: str) -> list[float]:
        if self.client is None:
            raise RuntimeError("для векторного поиска нужен openai_client")
        resp = self.client.embeddings.create(
            model=self.cfg["embedding"]["model"], input=[query])
        return resp.data[0].embedding

    def vector_search(self, query: str, k: int | None = None,
                      regulation: str | list[str] | None = None) -> list[tuple[str, float]]:
        k = k or self.cfg["search"]["candidates"]
        self._ensure_qdrant()
        flt = self._regulation_filter(regulation)
        hits = self._qdrant.query_points(
            collection_name=self.cfg["qdrant"]["collection"],
            query=self.embed_query(query), limit=k, query_filter=flt).points
        return [(h.payload["chunk_id"], float(h.score)) for h in hits]

    # --- Гибрид ---

    @staticmethod
    def _regulation_filter(regulation: str | list[str] | None):
        """Payload-фильтр Qdrant: MatchValue для одного регламента,
        MatchAny для списка (условие «regulation_id ∈ список»)."""
        allowed = as_regulation_list(regulation)
        if not allowed:
            return None
        if len(allowed) == 1:
            match = qm.MatchValue(value=allowed[0])
        else:
            match = qm.MatchAny(any=allowed)
        return qm.Filter(must=[qm.FieldCondition(key="regulation_id", match=match)])

    def hybrid_search(self, query: str, k: int | None = None,
                      regulation: str | list[str] | None = None) -> list[tuple[str, float]]:
        k = k or self.cfg["search"]["top_k"]
        n = self.cfg["search"]["candidates"]
        bm25_ids = [cid for cid, _ in self.bm25_search(query, n, regulation)]
        vec_ids = [cid for cid, _ in self.vector_search(query, n, regulation)]
        return rrf_fuse([bm25_ids, vec_ids], self.cfg["search"]["rrf_k"], k)
