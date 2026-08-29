"""Смоук query rewriting (НУЖЕН ключ OpenAI; в контейнере ~3 мин на сборку
Qdrant-коллекции в памяти).

Две задачи:
1. Прогреть кэш переформулировок (data/query_rewrites.json) для ВСЕХ
   запросов чек-листа — дальше вердикты и метрики работают без повторных
   вызовов CHEAP-модели, и у ревьюера те же переформулировки.
2. Контрольный кейс Дня 3: «Какие аллергены нужно указывать в маркировке?»
   BM25 его не находил (в 022 — «компоненты, вызывающие аллергические
   реакции»). Показываем место целевого пункта 022 ч.4.4 п.14 по четырём
   методам: BM25 / вектор / гибрид / гибрид+rewriting.

Запуск из корня:  python evaluation/smoke_rewrite.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from openai import OpenAI

from labelcheck.aspects import find_basis_chunks, load_aspects
from labelcheck.retrieval import Retriever
from labelcheck.rewrite import hybrid_search_rewritten, rewrite_query

ALLERGEN_QUERY = "Какие аллергены нужно указывать в маркировке?"
TARGET = {"reg": "ТР ТС 022/2011", "subsection": "4.4", "clauses": ["14"]}


def rank_of(ranking: list[str], targets: set[str]) -> str:
    for pos, cid in enumerate(ranking, start=1):
        if cid in targets:
            return f"место {pos}"
    return f"нет в топ-{len(ranking)}"


def main():
    load_dotenv()
    client = OpenAI()
    retriever = Retriever(openai_client=client)
    data = load_aspects()

    # 1. Прогрев кэша: все запросы всех регламентных аспектов
    print("Прогрев кэша переформулировок:")
    for aspect in data["aspects"]:
        for query in aspect.get("queries", []):
            rewrites = rewrite_query(query, client)
            print(f"  [{aspect['id']:>2}] «{query}»")
            for r in rewrites:
                print(f"        → «{r}»")

    # 2. Контрольный кейс «аллергены»
    targets = {c["chunk_id"] for c in find_basis_chunks(TARGET, retriever.chunks)}
    print(f"\nКейс «аллергены» → цель 022 ч.4.4 п.14 ({len(targets)} чанк):")
    k = 10
    bm25 = [cid for cid, _ in retriever.bm25_search(ALLERGEN_QUERY, k)]
    vec = [cid for cid, _ in retriever.vector_search(ALLERGEN_QUERY, k)]
    hyb = [cid for cid, _ in retriever.hybrid_search(ALLERGEN_QUERY, k)]
    hyb_rw = [cid for cid, _ in hybrid_search_rewritten(
        retriever, ALLERGEN_QUERY, k, client=client)]
    print(f"  BM25:              {rank_of(bm25, targets)}")
    print(f"  Вектор:            {rank_of(vec, targets)}")
    print(f"  Гибрид RRF:        {rank_of(hyb, targets)}")
    print(f"  Гибрид + rewriting: {rank_of(hyb_rw, targets)}")


if __name__ == "__main__":
    main()
