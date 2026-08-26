"""Смоук-тест чанков корпуса.

Проверяет, что после любой правки parse.py / cleanup.yaml ключевые нормы
на месте и структура чанков не деградировала.

Запуск из корня репозитория:  python tests/test_chunks.py
(совместим и с pytest: pytest tests/)
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"


def load_chunks() -> list[dict]:
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


CHUNKS = load_chunks()


def find(**conditions):
    """Есть ли чанк, у которого все указанные поля имеют указанные значения.
    Специальный ключ text_contains ищет подстроку в тексте чанка."""
    text_sub = conditions.pop("text_contains", None)
    sub_prefix = conditions.pop("subsection_prefix", None)
    for c in CHUNKS:
        if all(c.get(k) == v for k, v in conditions.items()) \
                and (text_sub is None or text_sub in c["text"]) \
                and (sub_prefix is None or (c.get("subsection") or "").startswith(sub_prefix)):
            return True
    return False


# --- Доменные проверки: ключевые нормы, которые обязаны быть в корпусе ---

def test_022_obligatory_info():
    """022 ч.4.1 п.1 — перечень обязательных сведений маркировки."""
    assert find(regulation_id="ТР ТС 022/2011", subsection_prefix="4.1", clause="1",
                text_contains="должна содержать следующие сведения")


def test_022_nutrition():
    """022 ч.4.9 — пищевая ценность."""
    assert find(regulation_id="ТР ТС 022/2011", subsection_prefix="4.9", clause="1",
                text_contains="нергетическую ценность")


def test_022_definitions():
    """022 ст.2 — определения (текст без номеров пунктов не потерян)."""
    assert find(regulation_id="ТР ТС 022/2011", text_contains="придуманное название")


def test_005_wrapped_clause():
    """005 п.11.2 — многоуровневый пункт с переносом строки (регресс фикса 1)."""
    assert find(regulation_id="ТР ТС 005/2011", clause="11.2")


def test_040_glaze():
    """040 п.33 — нормы массы глазури."""
    assert find(regulation_id="ТР ЕАЭС 040/2016", clause="33", text_contains="глазури")


def test_034_marking():
    """034 п.106 — требования к маркировке мясной продукции."""
    assert find(regulation_id="ТР ТС 034/2013", clause="106", text_contains="аркировка")


def test_029_new_additives():
    """029 прил.2 — добавки из Решения N 84 (Е243, Е1205): редакция актуальна."""
    assert find(appendix="2", regulation_id="ТР ТС 029/2012", text_contains="Е243")
    assert find(appendix="2", regulation_id="ТР ТС 029/2012", text_contains="Е1205")


def test_029_additive_function():
    """029 прил.2 — Е-код вместе со своим функциональным классом в одном чанке."""
    assert any(c.get("appendix") == "2" and "Е243" in c["text"]
               and "консервант" in c["text"] for c in CHUNKS)


def test_022_all_subsections():
    """022 ст.4 — все 12 подразделов распознаны (регресс: 4.8/4.10/4.11
    терялись из-за переноса длинного заголовка на вторую строку PDF)."""
    nums = {(c.get("subsection") or "").split(" ")[0].rstrip(".")
            for c in CHUNKS if c["regulation_id"] == "ТР ТС 022/2011"}
    expected = {f"4.{i}" for i in range(1, 13)}
    assert expected <= nums, f"нет подразделов: {sorted(expected - nums)}"


def test_022_manufacturer_in_48():
    """022 ч.4.8 п.1 — изготовитель (аспект 10) в правильном подразделе,
    а не в 4.7 (регресс потерянного заголовка)."""
    assert find(regulation_id="ТР ТС 022/2011", subsection_prefix="4.8", clause="1",
                text_contains="аименование и место нахождения изготовителя")


def test_022_gmo_in_411():
    """022 ч.4.11 п.1 — сведения о ГМО (аспект 12) в правильном подразделе."""
    assert find(regulation_id="ТР ТС 022/2011", subsection_prefix="4.11", clause="1",
                text_contains="ГМО")


def test_021_children_meat_clauses():
    """021 ст.8 п.8/п.9 — перечни мясного сырья для детского питания слиты
    в свои пункты (регресс: обрывок аннотации «115. - См...» рвал их
    на псевдо-пункты с номером решения ЕЭК вместо номера пункта)."""
    assert find(regulation_id="ТР ТС 021/2011", clause="8",
                text_contains="свинина жилованная")
    assert not find(regulation_id="ТР ТС 021/2011", clause="115")


def test_no_annotation_scrap_chunks():
    """Обрывки пометок Кодекса «NN. - См. предыдущую редакцию)» не становятся
    пунктами ни в одном документе (регресс)."""
    pat = re.compile(r"^\d+(?:\.\d+)*\.\s*-\s*См\. предыдущую редакцию")
    bad = [c["chunk_id"] for c in CHUNKS if pat.match(c["text"])]
    assert not bad, f"чанки-обрывки аннотаций: {bad[:5]}"


# --- Структурные проверки ---

def test_all_regulations_present():
    ids = {c["regulation_id"] for c in CHUNKS}
    assert len(ids) == 6, f"регламентов в чанках: {len(ids)}, ожидалось 6"


def test_chunk_sizes():
    assert max(len(c["text"]) for c in CHUNKS) <= 3500
    tiny = sum(1 for c in CHUNKS if len(c["text"]) < 120)
    assert tiny / len(CHUNKS) < 0.10, f"крошечных чанков {tiny} — больше 10%"


def test_required_fields():
    for c in CHUNKS:
        assert c["text"].strip(), f"пустой текст: {c['chunk_id']}"
        assert c["regulation_id"] and c["chunk_id"] and c["edition"]


def test_body_clause_address_unique():
    """Адрес пункта в теле документа (регламент + раздел + подраздел + пункт)
    уникален с точностью до part-разрезки длинного пункта. Это опора матчера
    ground truth: попадание в пункт считается по этому адресу."""
    groups = defaultdict(list)
    for c in CHUNKS:
        if c.get("clause") and not c.get("appendix"):
            groups[(c["regulation_id"], c.get("section"),
                    c.get("subsection"), c["clause"])].append(c)
    bad = {k: [c["chunk_id"] for c in v] for k, v in groups.items()
           if len(v) > 1 and not all(c.get("part") for c in v)}
    assert not bad, f"неуникальные адреса пунктов: {list(bad.items())[:3]}"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"✅ {name}: {fn.__doc__ or ''}".strip())
        except AssertionError as e:
            failed += 1
            print(f"❌ {name}: {e or fn.__doc__}")
    print(f"\n{len(tests) - failed}/{len(tests)} проверок пройдено")
    sys.exit(1 if failed else 0)
