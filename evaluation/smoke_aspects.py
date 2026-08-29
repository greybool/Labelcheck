"""Смоук аспектных запросов БЕЗ API: BM25 по каждому запросу чек-листа.

Зачёт (для глаз, не метрика): хоть один чанк-основание аспекта попал в
топ-10 BM25 при поиске по регламентам аспекта (категории включены обе —
максимальная ширина). Провал не значит «сломано»: часть оснований лексически
далека от запроса и достаётся векторной половиной гибрида или rewriting'ом —
формально это измерит День 7.

Запуск из корня:  python evaluation/smoke_aspects.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from labelcheck.aspects import find_basis_chunks, load_aspects, resolve_regulations
from labelcheck.retrieval import Retriever

TOP_K = 10


def address(chunk: dict) -> str:
    parts = [chunk["regulation_id"].replace("ТР ТС ", "").replace("ТР ЕАЭС ", "")]
    if chunk.get("appendix"):
        parts.append(f"прил.{chunk['appendix']}")
    if chunk.get("subsection"):
        parts.append(chunk["subsection"].split(".")[0] + "." + chunk["subsection"].split(".")[1])
    if chunk.get("clause"):
        parts.append(f"п.{chunk['clause']}")
    return " ".join(parts)


def main():
    retriever = Retriever()
    data = load_aspects()
    by_id = {c["chunk_id"]: c for c in retriever.chunks}

    total = hits = 0
    for aspect in data["aspects"]:
        if aspect["group"] != "regulatory":
            continue
        regs = resolve_regulations(aspect, {"meat", "fish"})
        basis_ids = set()
        for basis in aspect.get("basis", []):
            basis_ids |= {c["chunk_id"] for c in find_basis_chunks(basis, retriever.chunks)}

        print(f"\n=== {aspect['id']}. {aspect['name']} "
              f"[{', '.join(r.split()[-1] for r in regs)}] ===")
        for query in aspect["queries"]:
            total += 1
            top = [cid for cid, _ in retriever.bm25_search(query, TOP_K, regs)]
            hit = any(cid in basis_ids for cid in top)
            hits += hit
            mark = "✓" if hit else "✗"
            shown = " | ".join(address(by_id[cid]) for cid in top[:3])
            print(f"  {mark} «{query}»")
            print(f"      топ-3: {shown}")

    print(f"\nИтог: {hits}/{total} запросов находят основание своего аспекта "
          f"в топ-{TOP_K} одним лишь BM25")


if __name__ == "__main__":
    main()
