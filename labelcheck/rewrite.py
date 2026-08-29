"""Query rewriting: переформулировки запроса «языком регламента» (CHEAP_MODEL).

Зачем: язык упаковки и язык норматива расходятся — «аллергены» в 022
называются «компоненты, вызывающие аллергические реакции», и BM25 такой
запрос не находит (смоук Дня 3). Дешёвая модель делает 2 переформулировки,
и гибридный поиск сливает ранги по ВСЕМ выдачам: (BM25 + вектор) ×
(оригинал + переформулировки) через тот же rrf_fuse.

Принципы:
- оригинальный запрос всегда участвует — переформулировки только добавляют;
- кэш data/query_rewrites.json (в git): аспектные запросы статичны, API
  зовётся один раз, прогоны воспроизводимы у ревьюера;
- всё управление — labelcheck/config.yaml → rewrite (enabled, n, prompt).
"""

import hashlib
import json
import os
from pathlib import Path

from labelcheck.retrieval import ROOT, Retriever, load_config, rrf_fuse


def _cache_path(cfg: dict) -> Path:
    return ROOT / cfg["rewrite"]["cache"]


def _cache_key(model: str, prompt: str, query: str, n: int) -> str:
    """Ключ кэша: модель + промпт + запрос + число переформулировок.
    Меняется любой из них — переформулировки пересоздаются."""
    raw = "\x1f".join([model, prompt, query, str(n)])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _load_cache(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=1, sort_keys=True),
                   encoding="utf-8")
    tmp.replace(path)  # атомарная замена: не портим кэш при падении на записи


def rewrite_query(query: str, client=None, cfg: dict | None = None) -> list[str]:
    """Переформулировки запроса. [] — если rewriting выключен в конфиге.

    client (openai.OpenAI) нужен только при промахе кэша; запуск с тёплым
    кэшем работает без API (и без ключа).
    """
    cfg = cfg or load_config()
    rw = cfg["rewrite"]
    if not rw.get("enabled", False):
        return []

    model = os.environ[rw["model_env"]]
    prompt = rw["prompt"].format(n=rw["n"])
    key = _cache_key(model, prompt, query, rw["n"])

    path = _cache_path(cfg)
    cache = _load_cache(path)
    if key in cache:
        return cache[key]["rewrites"]

    if client is None:
        raise RuntimeError(
            f"переформулировок для запроса нет в кэше {path}, "
            "а openai-клиент не передан — либо передай client, "
            "либо выключи rewrite.enabled")

    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": prompt},
                  {"role": "user", "content": query}])
    data = json.loads(resp.choices[0].message.content)
    rewrites = [r.strip() for r in data.get("rewrites", [])
                if isinstance(r, str) and r.strip()][: rw["n"]]

    # В кэш кладём и исходный запрос — файл читаем глазами при разборе метрик.
    cache[key] = {"query": query, "model": model, "rewrites": rewrites}
    _save_cache(path, cache)
    return rewrites


def hybrid_search_rewritten(retriever: Retriever, query: str,
                            k: int | None = None,
                            regulation: str | list[str] | None = None,
                            client=None,
                            cfg: dict | None = None) -> list[tuple[str, float]]:
    """Гибрид с переформулировками: RRF по выдачам (BM25 + вектор) для
    оригинала и каждой переформулировки. При rewrite.enabled: false —
    в точности обычный hybrid_search (две выдачи вместо шести)."""
    cfg = cfg or retriever.cfg
    k = k or cfg["search"]["top_k"]
    n = cfg["search"]["candidates"]

    queries = [query] + rewrite_query(query, client, cfg)
    rankings = []
    for q in queries:
        rankings.append([cid for cid, _ in retriever.bm25_search(q, n, regulation)])
        rankings.append([cid for cid, _ in retriever.vector_search(q, n, regulation)])
    return rrf_fuse(rankings, cfg["search"]["rrf_k"], k)
