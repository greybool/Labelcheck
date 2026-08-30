"""Retrieval evaluation (День 7, ТЗ §5.1/§5.4): 6 прогонов + метрики.

Запуск из корня репозитория:  python evaluation/run_retrieval_eval.py
Пересчёт метрик из снапшотов:  python evaluation/run_retrieval_eval.py --metrics-only

Меряем ТРИ метода (BM25 / vector / hybrid RRF) × rewriting off/on = 6
прогонов на одном эталоне (evaluation/ground_truth.jsonl). Каждый прогон —
immutable-снапшот в evaluation/runs/<run_id>/ (meta.json + полные топ-10
выдачи по каждому вопросу); метрики (evaluation/metrics/) пересчитываемы
из снапшотов, расхождение пересчёта со старой метрикой — сигнал (§5.4).

Метрики — всегда все пять: Hit@1, Hit@3, Hit@5, Hit@10, MRR (средний
1/ранг в топ-10; цель вне топ-10 даёт 0). Разбивки: по категории чанка,
по стилю вопроса (verbatim/paraphrase — там виден эффект rewriting)
и по регламенту; в каждой ячейке — n.

Честность сравнения: поиск по ВСЕМУ корпусу без фильтров (боевые фильтры
аспектов — не для экзамена); hybrid — в боевой конфигурации (candidates,
rrf_k из config.yaml). Rewriting для метода X = RRF-слияние выдач X по
оригиналу и переформулировкам (для hybrid — ровно боевой
hybrid_search_rewritten). Эмбеддинги вопросов и переформулировки — в кэшах:
повторный запуск не зовёт API.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from evaluation.matcher import (CATEGORIES, STYLES, load_ground_truth,
                                rank_of_record, sha256_of)
from labelcheck import rewrite as RW
from labelcheck.retrieval import CHUNKS_PATH, Retriever, load_config, rrf_fuse

METHODS = ("bm25", "vector", "hybrid")
METRIC_KEYS = ("hit@1", "hit@3", "hit@5", "hit@10", "mrr")


# ── кэш эмбеддингов вопросов ─────────────────────────────────────────────────

def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def load_embedding_cache(path: Path, model: str) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=False)
    if str(data["model"]) != model:
        return {}  # сменилась модель — кэш недействителен, пересоздаём
    return {str(k): v for k, v in zip(data["keys"], data["vectors"])}


def save_embedding_cache(path: Path, cache: dict, model: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(cache)
    np.savez_compressed(path,
                        keys=np.asarray(keys),
                        vectors=np.asarray([cache[k] for k in keys],
                                           dtype=np.float32),
                        model=model)


def ensure_embeddings(texts: list[str], path: Path, ecfg: dict, client) -> tuple[dict, int]:
    """Дозаполняет кэш эмбеддингов недостающими текстами (батчами). → (кэш, токены)."""
    cache = load_embedding_cache(path, ecfg["model"])
    missing = sorted({t for t in texts if text_key(t) not in cache})
    tokens = 0
    if missing:
        print(f"Эмбеддинг {len(missing)} текстов (кэш: {len(cache)})…")
        for start in range(0, len(missing), ecfg["batch_size"]):
            batch = missing[start:start + ecfg["batch_size"]]
            resp = client.embeddings.create(model=ecfg["model"], input=batch)
            for item in sorted(resp.data, key=lambda d: d.index):
                cache[text_key(batch[item.index])] = np.asarray(item.embedding,
                                                               dtype=np.float32)
            tokens += resp.usage.total_tokens
        save_embedding_cache(path, cache, ecfg["model"])
    return cache, tokens


class CachedRetriever(Retriever):
    """Retriever, берущий эмбеддинги запросов из кэша вместо API.

    Прогоны детерминированы и бесплатны после разогрева кэша; запрос без
    эмбеддинга — громкая ошибка, а не тихий вызов API.
    """

    def __init__(self, config: dict, embeddings: dict[str, np.ndarray]):
        super().__init__(config)
        self._emb = embeddings

    def embed_query(self, query: str) -> list[float]:
        key = text_key(query)
        if key not in self._emb:
            raise KeyError(f"нет эмбеддинга в кэше для запроса {query[:80]!r}")
        return self._emb[key].tolist()


# ── переформулировки GT-вопросов ─────────────────────────────────────────────

def eval_rewrite_cfg(cfg: dict) -> dict:
    """Копия конфига с кэшем переформулировок ДЛЯ ЭТАЛОНА (не аспектным)."""
    rcfg = json.loads(json.dumps(cfg))  # глубокая копия
    rcfg["rewrite"]["enabled"] = True
    rcfg["rewrite"]["cache"] = cfg["eval"]["rewrites_cache"]
    return rcfg


def prefetch_rewrites(questions: list[str], rcfg: dict, client,
                      workers: int) -> int:
    """Параллельно докачивает недостающие переформулировки в кэш. → токены.

    rewrite_query пишет кэш атомарно, но при параллельных вызовах записи
    затирали бы друг друга — поэтому качаем сами и сохраняем кэш один раз.
    """
    rw = rcfg["rewrite"]
    model = os.environ[rw["model_env"]]
    prompt = rw["prompt"].format(n=rw["n"])
    path = RW._cache_path(rcfg)
    cache = RW._load_cache(path)
    missing = [q for q in questions
               if RW._cache_key(model, prompt, q, rw["n"]) not in cache]
    if not missing:
        return 0
    print(f"Переформулировки: {len(missing)} вопросов "
          f"(в кэше уже {len(cache)})…")

    tokens = 0

    def fetch(q: str):
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": prompt},
                              {"role": "user", "content": q}])
                data = json.loads(resp.choices[0].message.content)
                rewrites = [r.strip() for r in data.get("rewrites", [])
                            if isinstance(r, str) and r.strip()][: rw["n"]]
                return q, rewrites, resp.usage.total_tokens
            except Exception:  # noqa: BLE001
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(fetch, q) for q in missing]):
            q, rewrites, used = fut.result()
            key = RW._cache_key(model, prompt, q, rw["n"])
            cache[key] = {"query": q, "model": model, "rewrites": rewrites}
            tokens += used
            done += 1
            if done % 50 == 0:  # обрыв процесса не теряет оплаченные вызовы
                RW._save_cache(path, cache)
            print(f"\r  {done}/{len(missing)}", end="", flush=True)
    print()
    RW._save_cache(path, cache)
    return tokens


# ── поиск для одного вопроса ─────────────────────────────────────────────────

def run_search(retr: Retriever, method: str, queries: list[str],
               scfg: dict) -> list[str]:
    """Топ-10 chunk_id. queries = [оригинал] (+ переформулировки при rw on).

    Один запрос → нативная выдача метода (боевой режим без rewriting);
    несколько → RRF-слияние выдач метода по всем запросам (для hybrid это
    в точности схема боевого hybrid_search_rewritten: (BM25+вектор) × запросы).
    """
    cands, top_k, rrf_k = scfg["candidates"], scfg["top_k"], scfg["rrf_k"]
    if method == "hybrid":
        rankings = []
        for q in queries:
            rankings.append([cid for cid, _ in retr.bm25_search(q, cands)])
            rankings.append([cid for cid, _ in retr.vector_search(q, cands)])
        return [cid for cid, _ in rrf_fuse(rankings, rrf_k, top_k)]
    fn = retr.bm25_search if method == "bm25" else retr.vector_search
    if len(queries) == 1:
        return [cid for cid, _ in fn(queries[0], top_k)]
    rankings = [[cid for cid, _ in fn(q, cands)] for q in queries]
    return [cid for cid, _ in rrf_fuse(rankings, rrf_k, top_k)]


# ── снапшоты прогонов (immutable) ────────────────────────────────────────────

def write_run(runs_dir: Path, run_id: str, meta: dict,
              results: list[dict]) -> None:
    """Пишет снапшот во временную папку и атомарно переименовывает.

    Существующий run никогда не перезаписывается — иммутабельность (§5.4).
    """
    final = runs_dir / run_id
    if final.exists():
        raise FileExistsError(f"прогон {run_id} уже существует — снапшоты "
                              "иммутабельны, удаление только руками")
    tmp = runs_dir / f".tmp_{run_id}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    (tmp / "meta.json").write_text(json.dumps(meta, ensure_ascii=False,
                                              indent=1), encoding="utf-8")
    with open(tmp / "results.jsonl", "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.rename(final)


def read_run(run_dir: Path) -> tuple[dict, list[dict]]:
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    with open(run_dir / "results.jsonl", encoding="utf-8") as f:
        results = [json.loads(line) for line in f]
    return meta, results


# ── метрики ──────────────────────────────────────────────────────────────────

def metrics_of(ranks: list[int | None]) -> dict:
    """Все пять метрик + n по списку рангов (None = вне топ-10)."""
    n = len(ranks)
    out = {"n": n}
    for k in (1, 3, 5, 10):
        out[f"hit@{k}"] = round(sum(1 for r in ranks if r and r <= k) / n, 4) if n else 0.0
    out["mrr"] = round(sum(1 / r for r in ranks if r) / n, 4) if n else 0.0
    return out


def compute_metrics(records: list[dict], results: list[dict],
                    index: dict) -> dict:
    """Ранги по каждому вопросу → сводная + разбивки (категория/стиль/регламент)."""
    assert len(records) == len(results), "снапшот не совпадает с эталоном по длине"
    ranks, groups = [], defaultdict(list)
    for rec, row in zip(records, results):
        assert rec["chunk_id"] == row["chunk_id"], "снапшот рассинхронизирован"
        r = rank_of_record(rec, row["retrieved"], index)
        ranks.append(r)
        groups[("category", rec["category"])].append(r)
        groups[("style", rec.get("style", "verbatim"))].append(r)
        groups[("regulation", rec["regulation_id"])].append(r)
    out = {"overall": metrics_of(ranks)}
    for dim in ("category", "style", "regulation"):
        out[f"by_{dim}"] = {name: metrics_of(rs)
                            for (d, name), rs in sorted(groups.items())
                            if d == dim}
    return out


def md_table(rows: list[tuple[str, str, dict]]) -> str:
    """Markdown-таблица: (метод, rewriting, метрики) → строки."""
    lines = ["| метод | rewriting | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | n |",
             "|---|---|---|---|---|---|---|---|"]
    for method, rw, m in rows:
        cells = " | ".join(f"{m[k]:.3f}" for k in METRIC_KEYS)
        lines.append(f"| {method} | {rw} | {cells} | {m['n']} |")
    return "\n".join(lines)


def write_summary(metrics_dir: Path, all_metrics: dict[str, dict],
                  run_order: list[tuple[str, str, str]], gt_meta: dict,
                  gt_sha: str) -> None:
    """metrics/summary.md — все таблицы одним файлом (для README Дня 7)."""
    md = ["# Retrieval-метрики LabelCheck (День 7)", "",
          f"Эталон: {gt_meta['n_questions']} вопросов, модель-генератор "
          f"{gt_meta['model']}, corpus SHA256 `{gt_meta['corpus_sha256'][:12]}…`, "
          f"эталон SHA256 `{gt_sha[:12]}…`.", "",
          "Hit@K — доля вопросов, у которых правильный пункт в первых K "
          "результатах; MRR — средний 1/ранг в топ-10 (вне топ-10 = 0). "
          "Правила зачёта: evaluation/EVALUATION.md. Ячейки с n < 20 — "
          "шум, не сигнал.", "",
          "## Сводная (все вопросы)", ""]
    md.append(md_table([(m, rw, all_metrics[rid]["overall"])
                        for m, rw, rid in run_order]))

    for dim, title in (("style", "стилю вопроса"),
                       ("category", "категории чанка"),
                       ("regulation", "регламенту")):
        md += ["", f"## По {title}"]
        names = sorted({name for rid in all_metrics
                        for name in all_metrics[rid][f"by_{dim}"]})
        for name in names:
            n = next(all_metrics[rid][f"by_{dim}"][name]["n"]
                     for _, _, rid in run_order
                     if name in all_metrics[rid][f"by_{dim}"])
            warn = " ⚠ n<20" if n < 20 else ""
            md += ["", f"### {name}{warn}", ""]
            md.append(md_table([(m, rw, all_metrics[rid][f"by_{dim}"][name])
                                for m, rw, rid in run_order
                                if name in all_metrics[rid][f"by_{dim}"]]))
    (metrics_dir / "summary.md").write_text("\n".join(md) + "\n",
                                            encoding="utf-8")


# ── оркестрация ──────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-only", action="store_true",
                        help="только пересчитать метрики из снапшотов runs/")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    cfg = load_config()
    ecfg = cfg["eval"]
    gt_path, meta_path = ROOT / ecfg["gt_path"], ROOT / ecfg["gt_meta_path"]
    runs_dir, metrics_dir = ROOT / ecfg["runs_dir"], ROOT / ecfg["metrics_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Эталон: пломба корпуса сверяется внутри load_ground_truth (громкий отказ).
    records, index = load_ground_truth(gt_path, meta_path, CHUNKS_PATH)
    gt_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    gt_sha = sha256_of(gt_path)
    questions = [r["question"] for r in records]
    print(f"Эталон: {len(records)} вопросов, пломба корпуса сошлась.")

    rcfg = eval_rewrite_cfg(cfg)
    run_order = [(m, rw, f"{m}_rw-{rw}")
                 for m in METHODS for rw in ("off", "on")]

    drift = False
    if not args.metrics_only:
        from openai import OpenAI
        client = OpenAI()

        rw_tokens = prefetch_rewrites(questions, rcfg, client,
                                      ecfg["concurrency"])
        rewrites = {q: RW.rewrite_query(q, None, rcfg) for q in questions}

        all_texts = questions + [x for rs in rewrites.values() for x in rs]
        emb_cache, emb_tokens = ensure_embeddings(
            all_texts, ROOT / ecfg["question_embeddings_cache"],
            cfg["embedding"], client)
        print(f"API-расход: переформулировки {rw_tokens:,} ток., "
              f"эмбеддинги {emb_tokens:,} ток.")

        retr = CachedRetriever(cfg, emb_cache)
        for method, rw, run_id in run_order:
            if (runs_dir / run_id).exists():
                meta, _ = read_run(runs_dir / run_id)
                if meta["gt_sha256"] != gt_sha:
                    raise ValueError(f"прогон {run_id} сделан для другого "
                                     "эталона — перенеси/удали его руками")
                print(f"{run_id}: снапшот уже есть — поиск пропущен")
                continue
            t0 = time.time()
            results = []
            for i, rec in enumerate(records):
                queries = ([rec["question"]] + rewrites[rec["question"]]
                           if rw == "on" else [rec["question"]])
                results.append({"qi": i, "chunk_id": rec["chunk_id"],
                                "retrieved": run_search(retr, method, queries,
                                                        cfg["search"])})
                if (i + 1) % 100 == 0:
                    print(f"\r{run_id}: {i + 1}/{len(records)}",
                          end="", flush=True)
            meta = {"run_id": run_id, "method": method, "rewriting": rw,
                    "date": date.today().isoformat(),
                    "corpus_sha256": gt_meta["corpus_sha256"],
                    "gt_sha256": gt_sha,
                    "n_questions": len(records),
                    "search": cfg["search"],
                    "embedding_model": cfg["embedding"]["model"],
                    "rewrite_n": rcfg["rewrite"]["n"]}
            write_run(runs_dir, run_id, meta, results)
            print(f"\r{run_id}: {len(records)} вопросов за "
                  f"{time.time() - t0:.0f} с → runs/{run_id}/")

    # Метрики — всегда пересчитываются из снапшотов (§5.4).
    all_metrics = {}
    for method, rw, run_id in run_order:
        run_dir = runs_dir / run_id
        if not run_dir.exists():
            print(f"{run_id}: снапшота нет — пропускаю")
            continue
        meta, results = read_run(run_dir)
        if meta["gt_sha256"] != gt_sha:
            print(f"{run_id}: снапшот от другого эталона — пропускаю")
            continue
        m = compute_metrics(records, results, index)
        m["run_id"], m["gt_sha256"] = run_id, gt_sha
        out = metrics_dir / f"{run_id}.json"
        if out.exists():
            old = json.loads(out.read_text(encoding="utf-8"))
            if old != m:
                drift = True
                print(f"⚠ {run_id}: пересчёт метрик РАЗОШЁЛСЯ со старым "
                      f"{out.name} — разбираться, не игнорировать (§5.4)")
        out.write_text(json.dumps(m, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        all_metrics[run_id] = m

    if all_metrics:
        order = [(m, rw, rid) for m, rw, rid in run_order
                 if rid in all_metrics]
        write_summary(metrics_dir, all_metrics, order, gt_meta, gt_sha)
        print(f"Метрики: {len(all_metrics)} прогонов → {metrics_dir}/ "
              "(summary.md — сводные таблицы)")
        print("\nСводная:")
        for m, rw, rid in order:
            o = all_metrics[rid]["overall"]
            print(f"  {m:7s} rw-{rw:3s}  " +
                  "  ".join(f"{k}={o[k]:.3f}" for k in METRIC_KEYS))
    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
