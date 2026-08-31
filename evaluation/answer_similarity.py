"""Вторая LLM-метрика (День 8, ТЗ §5.2, подход 2): cosine similarity ответа.

Запуск из корня репозитория:  python evaluation/answer_similarity.py

«Эталонный ответ» на ground-truth вопрос — текст его целевого чанка;
«ответ системы» — текст чанка на 1-м месте выдачи. Метрика — косинус между
их эмбеддингами (векторы берутся из кэша корпуса data/embeddings.npz,
API не вызывается, стоимость $0).

Что показывает сверх Hit@K: даже когда формально промах (топ-1 ≠ цель),
высокий косинус означает «ответ по смыслу рядом» (соседнее окно той же
таблицы, дубль нормы в другом регламенте — классы из разбора промахов,
EVALUATION.md §6), а низкий — настоящий промах. Считается по всем шести
снапшотам прогонов; в git идут только числа.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.matcher import load_ground_truth, rank_of_record
from evaluation.run_retrieval_eval import METHODS, read_run
from labelcheck.retrieval import CHUNKS_PATH, load_config


def load_corpus_vectors(cfg: dict) -> dict[str, np.ndarray]:
    """chunk_id → нормированный вектор (косинус = скалярное произведение)."""
    data = np.load(ROOT / cfg["embedding"]["cache"], allow_pickle=False)
    vecs = data["vectors"].astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return {str(cid): vecs[i] for i, cid in enumerate(data["chunk_ids"])}


def run_similarity(records: list[dict], results: list[dict], index: dict,
                   vectors: dict[str, np.ndarray]) -> dict:
    """Косинусы топ-1↔цель одного прогона, отдельно для попаданий и промахов."""
    sims, sims_hit, sims_miss = [], [], []
    for rec, row in zip(records, results):
        top1 = row["retrieved"][0] if row["retrieved"] else None
        if top1 is None:
            continue
        cos = float(vectors[rec["chunk_id"]] @ vectors[top1])
        sims.append(cos)
        if rank_of_record(rec, row["retrieved"], index) is None:
            sims_miss.append(cos)
        else:
            sims_hit.append(cos)

    def agg(xs):
        if not xs:
            return {"n": 0}
        a = np.asarray(xs)
        return {"n": len(xs), "mean": round(float(a.mean()), 4),
                "median": round(float(np.median(a)), 4),
                "p10": round(float(np.percentile(a, 10)), 4)}

    return {"all": agg(sims), "when_hit@10": agg(sims_hit),
            "when_miss@10": agg(sims_miss)}


def main() -> int:
    cfg = load_config()
    ecfg = cfg["eval"]
    records, index = load_ground_truth(ROOT / ecfg["gt_path"],
                                       ROOT / ecfg["gt_meta_path"], CHUNKS_PATH)
    vectors = load_corpus_vectors(cfg)

    out = {"metric": "cosine(текст целевого чанка, текст топ-1 выдачи), "
                     "эмбеддинги text-embedding-3-small из кэша корпуса"}
    for m in METHODS:
        for rw in ("off", "on"):
            rid = f"{m}_rw-{rw}"
            run_dir = ROOT / ecfg["runs_dir"] / rid
            if not run_dir.exists():
                continue
            _, results = read_run(run_dir)
            out[rid] = run_similarity(records, results, index, vectors)
            a = out[rid]
            print(f"{rid:15s} все: {a['all']['mean']:.3f} | попадания: "
                  f"{a['when_hit@10'].get('mean', 0):.3f} | промахи: "
                  f"{a['when_miss@10'].get('mean', 0):.3f} "
                  f"(n промахов {a['when_miss@10']['n']})")

    path = ROOT / ecfg["similarity_metrics"]
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"→ {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
