"""Тесты слоя данных UI (блок C, день 9): SQLite-журнал, фидбек, правки
layout'а человеком + смоук самого Streamlit-приложения (AppTest, без API).

Запуск из корня репозитория:  python tests/test_store.py
(совместим и с pytest: pytest tests/)
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from labelcheck import store as S

REPORT = {
    "meta": {"source_pdf": "test.pdf", "source_sha256": "0" * 64,
             "categories": {"meat": ["свинин"]}, "category_scan": "targeted",
             "tokens": {"m": {"prompt": 10, "completion": 5, "calls": 2}},
             "seconds": 12.3},
    "verdicts": [
        {"id": 1, "name": "Наименование", "status": "возможное нарушение",
         "applicable": True},
        {"id": 2, "name": "Состав", "status": "требует ручной проверки",
         "applicable": True},
        {"id": 3, "name": "Аллергены", "status": "соответствует",
         "applicable": True},
        {"id": 11, "name": "Импортёр", "status": "соответствует",
         "applicable": False},
    ],
}

LAYOUT = {
    "meta": {"source_pdf": "test.pdf"},
    "regions": [
        {"id": "r1", "kind": "composition", "lang": "ru",
         "text": "Состав: мука, вода", "status": "требует ручной проверки",
         "status_reason": "стилизованный шрифт"},
    ],
}


def _con():
    tmp = tempfile.mkdtemp()
    return S.connect(Path(tmp) / "test.db")


def test_record_check_counts_statuses():
    """Счётчики статусов в журнале: нарушение/ручная/ок считаются только
    по применимым вердиктам, не применимые — отдельно."""
    con = _con()
    cid = S.record_check(con, REPORT, "data/reports/x.md")
    row = S.fetch_checks(con)[0]
    assert row["id"] == cid
    assert (row["n_violation"], row["n_manual"], row["n_ok"], row["n_na"]) == (1, 1, 1, 1)
    assert json.loads(row["categories"]) == ["meat"]
    assert row["report_path"].endswith("x.md")


def test_feedback_roundtrip_and_validation():
    """Фидбек пишется и читается; мусорные rating/note_type отклоняются."""
    con = _con()
    cid = S.record_check(con, REPORT)
    S.record_feedback(con, cid, 1, "Наименование", "возможное нарушение",
                      rating="up")
    S.record_feedback(con, cid, 2, "Состав", "требует ручной проверки",
                      rating="down", note="запросить вид сырья",
                      note_type="supplier")
    fb = S.fetch_feedback(con, cid)
    assert len(fb) == 2 and fb[1]["note_type"] == "supplier"
    for bad in ({"rating": "meh"}, {"note_type": "не тот тип"}):
        try:
            S.record_feedback(con, cid, 1, "x", "соответствует", **bad)
        except ValueError:
            continue
        raise AssertionError(f"ожидали ValueError на {bad}")


def test_export_notes_only_nonempty():
    """Выгрузка заметок отдаёт только записи с текстом (👍 без заметки —
    не заметка)."""
    con = _con()
    cid = S.record_check(con, REPORT)
    S.record_feedback(con, cid, 1, "a", "соответствует", rating="up")
    S.record_feedback(con, cid, 2, "b", "соответствует", note="дизайнеру: логотип",
                      note_type="designer")
    notes = S.export_notes(con, cid)
    assert len(notes) == 1 and notes[0]["aspect_id"] == 2


def test_apply_region_edit_keeps_history():
    """Правка человека: текст заменён, статус «прочитано», старый текст —
    в истории edits; повторная правка тем же текстом — no-op."""
    layout = json.loads(json.dumps(LAYOUT))
    changed = S.apply_region_edit(layout, "r1", "Состав: мука, вода, соль")
    r = layout["regions"][0]
    assert changed and r["text"].endswith("соль")
    assert r["status"] == "прочитано" and r["human_edited"]
    assert r["edits"][0]["was"] == "Состав: мука, вода"
    assert S.apply_region_edit(layout, "r1", "Состав: мука, вода, соль") is False
    try:
        S.apply_region_edit(layout, "нет-такого", "x")
    except KeyError:
        pass
    else:
        raise AssertionError("ожидали KeyError на неизвестный регион")


def test_save_layout_backs_up_original_once():
    """Первое сохранение правленого layout'а кладёт рядом .orig.json;
    последующие бэкап не трогают (канонический layout не теряется)."""
    tmp = Path(tempfile.mkdtemp())
    p = tmp / "m.json"
    p.write_text(json.dumps(LAYOUT, ensure_ascii=False), encoding="utf-8")
    layout = json.loads(p.read_text(encoding="utf-8"))
    S.apply_region_edit(layout, "r1", "правка 1")
    S.save_layout(layout, p)
    orig = tmp / "m.orig.json"
    assert orig.exists()
    assert "правка 1" not in orig.read_text(encoding="utf-8")
    S.apply_region_edit(layout, "r1", "правка 2")
    S.save_layout(layout, p)
    assert "правка 1" not in orig.read_text(encoding="utf-8")  # бэкап нетронут
    assert "правка 2" in p.read_text(encoding="utf-8")


def test_app_boots_headless():
    """Смоук UI: приложение поднимается в AppTest без исключений, все три
    вкладки и дисклеймер на месте. Тяжёлые пути (vision, вердикты) — за
    кнопками, поэтому старт лёгкий и без API."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "labelcheck" / "app.py"),
                          default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert at.title and "LabelCheck" in at.title[0].value
    joined = " ".join(c.value for c in at.caption)
    assert "за специалистом и юристом" in joined  # дисклеймер ТЗ
    # Навигация — состоянием, а не вкладками: кнопки «Далее» должны
    # переключать шаг (вкладки Streamlit из кода переключить нельзя).
    assert at.session_state["nav"] == "1 · Макет"


def test_next_button_switches_step():
    """Кнопка «Перейти к шагу 2» реально переводит на второй шаг."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "labelcheck" / "app.py"), default_timeout=90)
    at.run()
    at.session_state["layout"] = LAYOUT
    at.session_state["layout_path"] = "/tmp/x.json"
    at.run()
    nxt = [b for b in at.button if "шагу 2" in b.label]
    assert nxt, [b.label for b in at.button]
    nxt[0].click().run()
    assert at.session_state["nav"] == "2 · Проверка", at.session_state["nav"]
    assert not at.exception, at.exception


def test_feedback_autosaves_without_button():
    """Оценка сохраняется сама при изменении виджета — отдельной кнопки
    «сохранить» в интерфейсе нет (замечание Сергея 31.08)."""
    con = _con()
    cid = S.record_check(con, REPORT)
    S.upsert_feedback(con, cid, 1, "Наименование", "возможное нарушение",
                      rating="up", note_type="designer")
    S.upsert_feedback(con, cid, 1, "Наименование", "возможное нарушение",
                      rating="down", note="ложное срабатывание",
                      note_type="none")
    fb = S.fetch_feedback(con, cid)
    assert len(fb) == 1, fb                       # правка заменила запись
    assert fb[0]["rating"] == "down" and fb[0]["note_type"] == "none"


def test_confirm_region_without_edit():
    """R-08: человек подтверждает сомнительный блок без правки текста —
    статус «прочитано», причина «подтверждено человеком», текст не тронут,
    в истории — запись confirm; повтор на уже прочитанном — no-op."""
    layout = json.loads(json.dumps(LAYOUT))
    r = layout["regions"][0]
    assert S.confirm_region(layout, "r1") is True
    assert r["status"] == "прочитано" and r["status_reason"] == "подтверждено человеком"
    assert r["text"] == "Состав: мука, вода" and r.get("human_confirmed")
    assert r["edits"][-1]["action"] == "confirm"
    assert r["edits"][-1]["was_status"] == "требует ручной проверки"
    assert S.confirm_region(layout, "r1") is False
    try:
        S.confirm_region(layout, "нет-такого")
    except KeyError:
        pass
    else:
        raise AssertionError("ожидали KeyError на неизвестный регион")


def test_decisions_for_check_shape():
    """R-04: выборка решений по проверке — словарь по aspect_id с rating /
    note_type / note; пустой note_type читается как 'none'."""
    con = _con()
    cid = S.record_check(con, REPORT)
    S.upsert_feedback(con, cid, 1, "Наименование", "возможное нарушение",
                      rating="up", note_type="designer", note="переписать")
    S.upsert_feedback(con, cid, 2, "Состав", "требует ручной проверки",
                      rating=None, note_type=None)
    d = S.decisions_for_check(con, cid)
    assert d[1] == {"rating": "up", "note_type": "designer", "note": "переписать"}
    assert d[2] == {"rating": None, "note_type": "none", "note": ""}
    assert S.decisions_for_check(con, cid + 99) == {}


# ── AppTest: интерфейс по замечаниям приёмки 02.09 ───────────────────────────

UI_REPORT = {
    "meta": {"source_pdf": "test.pdf", "source_sha256": "0" * 64,
             "categories": {}, "category_scan": "manual",
             "tokens": {}, "seconds": 3.0},
    "verdicts": [
        {"id": 1, "name": "Наименование продукции", "status": "возможное нарушение",
         "applicable": True, "explanation": "Нет вида сырья.", "citations": []},
        {"id": 3, "name": "Аллергены", "status": "соответствует",
         "applicable": True, "explanation": "Указаны.", "citations": []},
    ],
    "other_remarks": [
        {"id": 19, "key": "spelling", "name": "Орфография и пунктуация RU",
         "items": ["опечатка А", "опечатка Б", "опечатка В"]},
    ],
    "vision": {"missing": [], "text_layer_coverage": 0.9, "manual_regions": []},
}


def _app_with_tmp_db():
    """AppTest с базой во временной папке: приложение берёт connect из
    labelcheck.store, подменяем его до запуска, чтобы тесты не писали
    в рабочую data/labelcheck.db."""
    from streamlit.testing.v1 import AppTest
    tmp_db = Path(tempfile.mkdtemp()) / "ui.db"
    real_connect = S.connect
    S.connect = lambda path=None: real_connect(tmp_db)
    at = AppTest.from_file(str(ROOT / "labelcheck" / "app.py"), default_timeout=90)
    return at, tmp_db, real_connect


def test_step2_has_no_auto_category_mode():
    """R-05 (решение Сергея 02.09): автоопределение профильных регламентов
    убрано — на шаге 2 нет переключателя режимов, три переключателя
    видов продукции видны всегда, по умолчанию выключены."""
    at, _, real_connect = _app_with_tmp_db()
    try:
        at.run()
        at.session_state["layout"] = LAYOUT
        at.session_state["layout_path"] = "/tmp/x.json"
        at.session_state["nav"] = "2 · Проверка"
        at.run()
        assert not at.exception, at.exception
        labels = " ".join(r.label for r in at.radio)
        assert "автоматически" not in labels.lower(), labels
        toggles = [t for t in at.toggle if str(t.key).startswith("cat_")]
        assert len(toggles) == 3 and not any(t.value for t in toggles)
        texts = " ".join(m.value for m in at.markdown)
        assert "Базовые регламенты" in texts and "ТР ТС 022/2011" in texts
    finally:
        S.connect = real_connect


def test_decisions_survive_step_switch():
    """R-04 (обязательное замечание): оценка и решение, поставленные на шаге
    2, переживают уход на шаг 1 и возврат — поднимаются из базы, куда их
    положило автосохранение. Ключи виджетов Streamlit при этом стирает."""
    at, tmp_db, real_connect = _app_with_tmp_db()
    try:
        at.run()
        con = real_connect(tmp_db)
        cid = S.record_check(con, UI_REPORT)
        con.close()
        at.session_state["layout"] = LAYOUT
        at.session_state["layout_path"] = "/tmp/x.json"
        at.session_state["report"] = UI_REPORT
        at.session_state["plan"] = []
        at.session_state["check_id"] = cid
        at.session_state["nav"] = "2 · Проверка"
        at.run()
        assert not at.exception, at.exception
        # решение «запросить у поставщика» (индекс 2) и оценка 👍 по аспекту 1
        at.selectbox(key="dec_1").select_index(2).run()
        at.radio(key="rate_1").set_value("👍 верно").run()
        assert not at.exception, at.exception
        con = real_connect(tmp_db)
        saved = S.decisions_for_check(con, cid)
        con.close()
        assert saved[1]["note_type"] == "supplier" and saved[1]["rating"] == "up", saved
        # уходим на шаг 1 — Streamlit стирает ключи невидимых виджетов
        at.session_state["nav"] = "1 · Макет"
        at.run()
        assert "dec_1" not in at.session_state, "ключ должен быть стёрт — иначе тест не проверяет баг"
        # возвращаемся — значения восстановлены из базы
        at.session_state["nav"] = "2 · Проверка"
        at.run()
        assert not at.exception, at.exception
        assert at.session_state["dec_1"] == "supplier", at.session_state["dec_1"]
        assert at.session_state["rate_1"] == "👍 верно", at.session_state["rate_1"]
    finally:
        S.connect = real_connect


def test_confirm_button_on_step1():
    """R-08 в интерфейсе: у сомнительного блока есть кнопка «подтвердить»,
    клик переводит его в «прочитано» и сохраняет layout."""
    at, _, real_connect = _app_with_tmp_db()
    try:
        at.run()
        lp = Path(tempfile.mkdtemp()) / "m.json"
        lp.write_text(json.dumps(LAYOUT, ensure_ascii=False), encoding="utf-8")
        at.session_state["layout"] = json.loads(json.dumps(LAYOUT))
        at.session_state["layout_path"] = str(lp)
        at.run()
        btn = [b for b in at.button if "подтвердить" in b.label]
        assert btn and not btn[0].disabled, [(b.label, b.disabled) for b in at.button]
        btn[0].click().run()
        assert not at.exception, at.exception
        saved = json.loads(lp.read_text(encoding="utf-8"))
        r = saved["regions"][0]
        assert r["status"] == "прочитано" and r["status_reason"] == "подтверждено человеком"
        # после подтверждения кнопка гаснет
        at.run()
        btn = [b for b in at.button if "подтвердить" in b.label]
        assert btn and btn[0].disabled
    finally:
        S.connect = real_connect


def test_step1_layer_diff_table_and_coverage_warning():
    """R-12/R-15/R-24 в интерфейсе: у блока с подменой слова раскрыта
    таблица сверки (подмена, непрочитанные слова слоя, слова вне слоя);
    при низком покрытии слоя — список непрочитанных слов страницы; при
    «смеси кривых» — предупреждение."""
    at, _, real_connect = _app_with_tmp_db()
    try:
        at.run()
        layout = json.loads(json.dumps(LAYOUT))
        layout["regions"][0].update({
            "has_layer": True, "layer_partial": False,
            "word_substitutions": [{"layer": "молодой", "vision": "молотый",
                                    "kind": "substitution"}],
            "layer_missing_words": ["молодой", "варить"],
            "invented_words": ["молотый", "Exporter"], "invented_digits": ["4820140240955"],
            "status_reason": "ВОЗМОЖНАЯ ПОДМЕНА СЛОВА: на макете «молодой», прочитано «молотый»"})
        layout["text_layer_coverage"] = 0.82
        layout["unread_layer_words"] = ["молодой", "варить", "плите"]
        layout["text_layer_partial"] = True
        layout["text_layer_invented_share"] = 0.41
        lp = Path(tempfile.mkdtemp()) / "m.json"
        lp.write_text(json.dumps(layout, ensure_ascii=False), encoding="utf-8")
        at.session_state["layout"] = layout
        at.session_state["layout_path"] = str(lp)
        at.run()
        assert not at.exception, at.exception
        page = " ".join(m.value for m in at.markdown)
        assert "варить" in page and "Exporter, 4820140240955" in page, page[-600:]
        assert "не прочитано (1)" in page                    # «молодой» ушёл в пару
        assert "в текстовом слое PDF нет (2)" in page
        assert any("молодой" in str(t.value) for t in at.table)   # таблица подмен
        exp = [e.label for e in at.expander]
        assert any("расхождений: 4" in e for e in exp), exp
        assert any("Не прочитано 3 слов" in e for e in exp), exp
        warns = " ".join(w.value for w in at.warning)
        assert "в кривых" in warns and "41%" in warns
    finally:
        S.connect = real_connect


def test_plan_keeps_every_other_remark():
    """R-06: прочие замечания (орфография) идут в план каждым пунктом, а не
    первыми двумя; 👎 по блоку убирает их из плана."""
    at, tmp_db, real_connect = _app_with_tmp_db()
    try:
        at.run()
        con = real_connect(tmp_db)
        cid = S.record_check(con, UI_REPORT)
        con.close()
        for k, v in (("layout", LAYOUT), ("layout_path", "/tmp/x.json"),
                     ("report", UI_REPORT), ("plan", []), ("check_id", cid),
                     ("dec_19", "designer"), ("nav", "3 · План работ")):
            at.session_state[k] = v
        at.run()
        assert not at.exception, at.exception
        page = " ".join(m.value for m in at.markdown)
        for needle in ("опечатка А", "опечатка Б", "опечатка В"):
            assert needle in page, needle
        at.session_state["rate_19"] = "👎 система ошиблась"
        at.run()
        page = " ".join(m.value for m in at.markdown)
        assert "опечатка А" not in page
    finally:
        S.connect = real_connect


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
