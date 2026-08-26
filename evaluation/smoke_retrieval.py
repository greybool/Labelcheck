"""Смоук-тест ретрива глазами: три метода бок о бок на доменных вопросах.

Запуск из корня репозитория:  python evaluation/smoke_retrieval.py
Требует ключ OpenAI (.env) и кэш векторов (python ingestion/index.py).

Вопросы — в evaluation/smoke_questions.yaml. Это не метрика, а проверка
на грубые провалы до формальной оценки Дня 7.
"""

import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from labelcheck.retrieval import Retriever

TOP_SHOW = 5  # сколько строк выдачи печатать на метод


def ref(chunk: dict) -> str:
    """Короткая ссылка на чанк: регламент + подраздел/приложение + пункт."""
    reg = chunk["regulation_id"].replace("ТР ТС ", "").replace("ТР ЕАЭС ", "")
    place = ""
    if chunk.get("appendix"):
        place = f" прил.{chunk['appendix']}"
    elif chunk.get("subsection"):
        place = f" ч.{chunk['subsection'].split(' ')[0].rstrip('.')}"
    clause = f" п.{chunk['clause']}" if chunk.get("clause") else ""
    part = f"/{chunk['part']}" if chunk.get("part") else ""
    return f"{reg}{place}{clause}{part}"


def main() -> int:
    load_dotenv(ROOT / ".env")
    retriever = Retriever(openai_client=OpenAI())
    by_id = {c["chunk_id"]: c for c in retriever.chunks}
    questions = yaml.safe_load(
        open(ROOT / "evaluation" / "smoke_questions.yaml", encoding="utf-8"))["questions"]

    for item in questions:
        q = item["q"]
        print(f"\n{'=' * 78}\nВОПРОС: {q}\n(ожидаем: {item.get('expect', '—')})")
        results = {
            "BM25  ": retriever.bm25_search(q, TOP_SHOW),
            "vector": retriever.vector_search(q, TOP_SHOW),
            "hybrid": retriever.hybrid_search(q, TOP_SHOW),
        }
        for name, hits in results.items():
            refs = ", ".join(ref(by_id[cid]) for cid, _ in hits) or "(пусто)"
            print(f"  {name}: {refs}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
