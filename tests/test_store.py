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
                      note_type="на перепрогон")
    fb = S.fetch_feedback(con, cid)
    assert len(fb) == 2 and fb[1]["note_type"] == "на перепрогон"
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
                      note_type="замечание дизайнеру")
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
    assert len(at.tabs) == 3


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
