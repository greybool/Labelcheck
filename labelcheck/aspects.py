"""Аспектный чек-лист: загрузка aspects.yaml, детект категорий, адреса основ.

Три задачи модуля:

1. load_aspects() — прочитать и слегка проверить labelcheck/aspects.yaml
   (полная проверка структуры — tests/test_aspects.py).

2. detect_categories() — по тексту макета (наименование + состав, прочитанные
   vision-модулем) определить категории продукта (meat / fish) через
   слова-маркеры из конфига. Сравнение по ОСНОВЕ слова: маркеры и текст
   прогоняются через тот же стеммер, что и BM25, поэтому «свинины» находит
   маркер «свинина». Детект только расширяет поиск — применимость регламента
   к продукту решает вердикт (День 6).

3. resolve_regulations() / find_basis_chunks() — какие регламенты ищет аспект
   при данных категориях и какие чанки корпуса соответствуют каждому адресу
   из basis. Второе используется автотестом («выдуманный пункт роняет тест»)
   и Днём 6 (показ основания).
"""

from pathlib import Path

import yaml

from labelcheck.retrieval import Tokenizer, load_config

ASPECTS_PATH = Path(__file__).parent / "aspects.yaml"


def load_aspects(path: Path = ASPECTS_PATH) -> dict:
    """Читает aspects.yaml; базовая проверка, что ключевые поля на месте."""
    data = yaml.safe_load(open(path, encoding="utf-8"))
    for field in ("version", "category_markers", "aspects"):
        if field not in data:
            raise ValueError(f"aspects.yaml: нет обязательного поля {field!r}")
    return data


class CategoryDetector:
    """Определяет категории продукта по словам-маркерам (сравнение по стемам).

    markers: {"meat": [слова], "fish": [слова]} из aspects.yaml.
    Токенизатор — тот же, что в BM25 (lowercase, стеммер, нормализация
    Е-кодов), поэтому падежные формы маркеров перечислять не нужно.
    """

    def __init__(self, markers: dict[str, list[str]], tokenizer: Tokenizer | None = None):
        self.tokenizer = tokenizer or Tokenizer(load_config()["bm25"])
        self.stems = {
            category: set(self.tokenizer(" ".join(words)))
            for category, words in markers.items()
        }

    def __call__(self, text: str) -> dict[str, list[str]]:
        """Возвращает {категория: [сработавшие стемы]} — только непустые.

        Сработавшие стемы идут в отчёт: видно, ПОЧЕМУ подтянут регламент.
        """
        text_stems = set(self.tokenizer(text))
        hits = {}
        for category, marker_stems in self.stems.items():
            matched = sorted(marker_stems & text_stems)
            if matched:
                hits[category] = matched
        return hits


def resolve_regulations(aspect: dict, categories: set[str]) -> list[str]:
    """Итоговый список регламентов аспекта: база + категорийные по детекту.

    Порядок сохраняется (база первой), дубли убираются. Для аспектов
    group=other возвращает пустой список — они в регламенты не ходят.
    """
    result = list(aspect.get("regulations", []))
    extra = aspect.get("category_regulations", {})
    for category in sorted(categories):
        for reg in extra.get(category, []):
            if reg not in result:
                result.append(reg)
    return result


def basis_matches_chunk(basis: dict, chunk: dict) -> bool:
    """Соответствует ли чанк корпуса адресу basis из aspects.yaml.

    Поля адреса (все, кроме reg, необязательны; заданные должны совпасть все):
      reg        — regulation_id как в chunks.jsonl;
      subsection — часть статьи 022 («4.3» матчит подраздел «4.3. …»);
      article    — номер статьи («9» матчит секцию «Статья 9. …»;
                   «6» НЕ матчит «СТАТЬЯ 6_1.» — точка после номера);
      section    — римский раздел 034/040 («XI» матчит «XI. …», но не «XII.»);
      appendix   — номер приложения;
      clauses    — список номеров пунктов (части разрезанного пункта равнозначны).
    """
    if chunk["regulation_id"] != basis["reg"]:
        return False
    if "appendix" in basis:
        if str(chunk.get("appendix") or "") != str(basis["appendix"]):
            return False
    if "subsection" in basis:
        sub = chunk.get("subsection") or ""
        if not sub.startswith(f"{basis['subsection']}."):
            return False
    if "article" in basis:
        section = (chunk.get("section") or "").upper()
        if not section.startswith(f"СТАТЬЯ {basis['article']}."):
            return False
    if "section" in basis:
        section = chunk.get("section") or ""
        if not section.startswith(f"{basis['section']}."):
            return False
    if "clauses" in basis:
        if str(chunk.get("clause")) not in [str(c) for c in basis["clauses"]]:
            return False
    return True


def find_basis_chunks(basis: dict, chunks: list[dict]) -> list[dict]:
    """Все чанки корпуса, подпадающие под адрес basis."""
    return [c for c in chunks if basis_matches_chunk(basis, c)]
