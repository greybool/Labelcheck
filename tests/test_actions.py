"""Тесты плана работ (день 9): короткие пункты «что делать» без цитат
регламентов, правки человека, выгрузка в Markdown и Word. Без API —
LLM подменяется фейковым клиентом.

Запуск из корня репозитория:  python tests/test_actions.py
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import os

os.environ.setdefault("CHEAP_MODEL", "test-cheap")
os.environ.setdefault("MAIN_MODEL", "test-main")

from labelcheck import actions as A
from labelcheck.retrieval import load_config

CFG = load_config()

REPORT = {
    "meta": {"source_pdf": "test.pdf"},
    "verdicts": [
        {"id": 1, "name": "Наименование продукции", "status": "возможное нарушение",
         "applicable": True,
         "explanation": "В наименовании не указан вид сырья. Добавить вид мяса птицы."},
        {"id": 2, "name": "Состав", "status": "требует ручной проверки",
         "applicable": True,
         "explanation": "Запись «мясо курицы» не даёт понять вид сырья. "
                        "Запросить у производителя: бескостное мясо или мехобвалка."},
        {"id": 17, "name": "Размер шрифта", "status": "требует ручной проверки",
         "applicable": True,
         "explanation": "Автоматический замер не выполняется. Проверить высоту "
                        "букв линейкой."},
        {"id": 3, "name": "Аллергены", "status": "соответствует",
         "applicable": True, "explanation": "Аллергены указаны."},
        {"id": 11, "name": "Импортёр", "status": "соответствует",
         "applicable": False, "explanation": "Признаков импорта нет."},
    ],
}


class _Usage:
    prompt_tokens = 10
    completion_tokens = 5


class FakeClient:
    def __init__(self, payload, fail=False):
        self.calls = []
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                if fail:
                    raise RuntimeError("сеть недоступна")

                class _Resp:
                    usage = _Usage()

                    class _Choice:
                        class message:
                            content = json.dumps(payload, ensure_ascii=False)
                    choices = [_Choice()]
                return _Resp()

        class _Chat:
            completions = _Completions()
        self.chat = _Chat()


GOOD = {"items": [
    {"aspect_id": 1, "target": "designer",
     "text": "Добавить в наименование вид мяса птицы: «из мяса кур»."},
    {"aspect_id": 2, "target": "supplier",
     "text": "Запросить вид куриного сырья: бескостное мясо или мехобвалка."},
    {"aspect_id": 17, "target": "manual",
     "text": "Замерить высоту шрифта обязательных надписей."},
]}


def _cfg_tmp():
    """Конфиг с временным кэшем — тесты не пишут в рабочий data/."""
    import copy
    cfg = copy.deepcopy(CFG)
    cfg["actions"]["cache"] = str(Path(tempfile.mkdtemp()) / "actions.json")
    return cfg


def test_only_actionable_verdicts_enter_plan():
    """В план идут только нарушения и ручные проверки: «соответствует» и
    «не применимо» действий не требуют."""
    plan = A.build_plan(REPORT, client=FakeClient(GOOD), cfg=_cfg_tmp())
    assert {i["aspect_id"] for i in plan} == {1, 2, 17}
    assert all("текст" not in i for i in plan if not i["text"])


def test_plan_grouped_by_target():
    """Три адресата: дизайнеру, поставщику, проверить самому."""
    plan = A.build_plan(REPORT, client=FakeClient(GOOD), cfg=_cfg_tmp())
    by_id = {i["aspect_id"]: i["target"] for i in plan}
    assert by_id == {1: "designer", 2: "supplier", 17: "manual"}
    assert [i["target"] for i in plan] == ["designer", "supplier", "manual"]


def test_lost_aspect_recovered_by_fallback():
    """Аспект, который модель молча потеряла, добирается кодом: пропущенное
    замечание хуже неудачной формулировки."""
    partial = {"items": [GOOD["items"][0]]}
    plan = A.build_plan(REPORT, client=FakeClient(partial), cfg=_cfg_tmp())
    assert {i["aspect_id"] for i in plan} == {1, 2, 17}
    recovered = [i for i in plan if i["aspect_id"] == 2][0]
    assert recovered["source"] == "fallback" and recovered["text"]


def test_api_failure_falls_back_not_raises():
    """Сбой вызова не роняет отчёт — план собирается без LLM."""
    plan = A.build_plan(REPORT, client=FakeClient(GOOD, fail=True), cfg=_cfg_tmp())
    assert {i["aspect_id"] for i in plan} == {1, 2, 17}
    assert all(i["source"] == "fallback" for i in plan)


def test_cache_second_call_free():
    """Повторный план по тому же отчёту берётся из кэша (API не зовём)."""
    cfg = _cfg_tmp()
    c1 = FakeClient(GOOD)
    A.build_plan(REPORT, client=c1, cfg=cfg)
    assert len(c1.calls) == 1
    c2 = FakeClient(GOOD)
    A.build_plan(REPORT, client=c2, cfg=cfg)
    assert c2.calls == []


def test_human_decisions_override_plan():
    """👎 «система ошиблась» убирает пункт; заметка человека заменяет текст;
    выбранный адресат перекрывает предложенный моделью."""
    plan = A.build_plan(REPORT, client=FakeClient(GOOD), cfg=_cfg_tmp())
    out = A.apply_human_decisions(plan, {
        1: {"rating": "down"},                       # ложное срабатывание
        2: {"rating": "up", "target": "designer",
            "note": "Указать вид сырья прямо в составе"},
        17: {"rating": None, "target": "none"},      # снято
    })
    assert {i["aspect_id"] for i in out} == {2}
    item = out[0]
    assert item["target"] == "designer" and item["edited_by_human"]
    assert item["text"] == "Указать вид сырья прямо в составе"


def test_markdown_has_no_regulation_citations():
    """Готовый документ — без номеров пунктов и «согласно ТР ТС»
    (требование Сергея: коротко и понятно, для дизайнера и поставщика)."""
    plan = A.build_plan(REPORT, client=FakeClient(GOOD), cfg=_cfg_tmp())
    md = A.render_plan_markdown(plan, REPORT)
    for needle in ("Замечания дизайнеру", "Запросить у поставщика",
                   "Проверить самостоятельно", "test.pdf"):
        assert needle in md, needle
    low = md.lower()
    assert "тр тс" not in low and "п." not in low.replace("pdf", "")


def test_docx_written():
    """Word-версия плана создаётся и не пуста."""
    plan = A.build_plan(REPORT, client=FakeClient(GOOD), cfg=_cfg_tmp())
    out = Path(tempfile.mkdtemp()) / "plan.docx"
    A.plan_to_docx(plan, REPORT, out)
    assert out.exists() and out.stat().st_size > 5000

    from docx import Document
    text = "\n".join(p.text for p in Document(str(out)).paragraphs)
    assert "Запросить вид куриного сырья" in text


def test_several_findings_per_aspect_all_kept():
    """R-06: несколько находок в одном аспекте — несколько пунктов плана.
    Раньше дедуп по aspect_id оставлял первый и молча терял остальные.
    Дословный повтор одного текста — единственное, что режется."""
    multi = {"items": [
        {"aspect_id": 1, "target": "designer", "text": "Добавить вид мяса птицы."},
        {"aspect_id": 1, "target": "designer", "text": "Убрать слово «натуральный»."},
        {"aspect_id": 1, "target": "designer", "text": "Убрать слово «натуральный»."},
        {"aspect_id": 2, "target": "supplier", "text": "Запросить вид сырья."},
        {"aspect_id": 17, "target": "manual", "text": "Замерить шрифт."},
    ]}
    plan = A.build_plan(REPORT, client=FakeClient(multi), cfg=_cfg_tmp())
    texts_1 = [i["text"] for i in plan if i["aspect_id"] == 1]
    assert texts_1 == ["Добавить вид мяса птицы.", "Убрать слово «натуральный»."], texts_1
    assert all(i["source"] == "llm" for i in plan)   # fallback не понадобился


def test_explanation_not_truncated_in_prompt():
    """R-06: объяснение уходит в модель целиком — обрезка до 900 символов
    теряла находки из хвоста."""
    long_tail = {**REPORT, "verdicts": [{
        "id": 1, "name": "Наименование продукции", "status": "возможное нарушение",
        "applicable": True,
        "explanation": "х" * 1200 + " ХВОСТ_НАХОДКА."}]}
    c = FakeClient(GOOD)
    A.build_plan(long_tail, client=c, cfg=_cfg_tmp())
    sent = c.calls[0]["messages"][1]["content"]
    assert "ХВОСТ_НАХОДКА" in sent


def test_human_note_collapses_aspect_to_one_item():
    """R-06: своя формулировка человека заменяет ВСЕ пункты аспекта одним;
    👎 убирает все пункты аспекта."""
    plan = [
        {"aspect_id": 1, "aspect_name": "Наименование", "target": "designer",
         "text": "пункт А", "source": "llm"},
        {"aspect_id": 1, "aspect_name": "Наименование", "target": "designer",
         "text": "пункт Б", "source": "llm"},
        {"aspect_id": 2, "aspect_name": "Состав", "target": "supplier",
         "text": "пункт В", "source": "llm"},
        {"aspect_id": 2, "aspect_name": "Состав", "target": "supplier",
         "text": "пункт Г", "source": "llm"},
    ]
    out = A.apply_human_decisions(plan, {
        1: {"rating": "up", "note": "Переписать наименование целиком"},
        2: {"rating": "down"},
    })
    assert [i["text"] for i in out] == ["Переписать наименование целиком"], out
    assert out[0]["edited_by_human"]
    # без заметки — все пункты обоих аспектов остаются
    out2 = A.apply_human_decisions(plan, {1: {"rating": "up"}, 2: {"rating": "up"}})
    assert [i["text"] for i in out2] == ["пункт А", "пункт Б", "пункт В", "пункт Г"]


def test_empty_plan_when_nothing_to_do():
    """Нет нарушений и ручных проверок — план пуст, документ это говорит."""
    clean = {"meta": {"source_pdf": "clean.pdf"},
             "verdicts": [{"id": 3, "name": "Аллергены", "status": "соответствует",
                           "applicable": True, "explanation": "ок"}]}
    plan = A.build_plan(clean, client=FakeClient(GOOD), cfg=_cfg_tmp())
    assert plan == []
    assert "не найдено" in A.render_plan_markdown(plan, clean)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"✅ {name}: {fn.__doc__ or ''}".strip())
        except Exception as e:
            failed += 1
            print(f"❌ {name}: {e or fn.__doc__}")
    print(f"\n{len(tests) - failed}/{len(tests)} проверок пройдено")
    sys.exit(1 if failed else 0)
