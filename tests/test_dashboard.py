"""Тесты дашборда мониторинга (День 10): нормализация журнала, стоимость,
разбивки по аспектам, метрики зрения из layout'ов, демо-копия базы без
личных данных + смоук шага «4 · Мониторинг» в AppTest (без API).

Запуск из корня репозитория:  python tests/test_dashboard.py
(совместим и с pytest: pytest tests/)
"""

import json
import re
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from labelcheck import dashboard as D  # noqa: E402
from labelcheck import store as S  # noqa: E402
from labelcheck.retrieval import load_config  # noqa: E402

CFG = load_config()

FULL_TOKENS = {"gpt-5.4-2026-03-05": {"prompt": 200_000, "completion": 30_000, "calls": 32},
               "gpt-5.4-mini-2026-03-17": {"prompt": 1_000, "completion": 100, "calls": 1}}
CACHED_TOKENS = {"gpt-5.4-mini-2026-03-17": {"prompt": 1_266, "completion": 90, "calls": 1}}


def _report(sha: str, statuses: dict, name="/Users/private/Секретный макет.pdf",
            tokens=None, scan="manual"):
    verdicts = [{"id": aid, "name": f"Аспект {aid}", "status": st, "applicable": True,
                 "citations": [{"chunk_id": "x"}]} for aid, st in statuses.items()]
    verdicts.append({"id": 11, "name": "Импортёр", "status": D.STATUS_COMPLIANT,
                     "applicable": False, "citations": []})
    return {"meta": {"source_pdf": name, "source_sha256": sha, "categories": {},
                     "category_scan": scan, "tokens": tokens or FULL_TOKENS,
                     "seconds": 300.0},
            "verdicts": verdicts}


def _tmp_con():
    return S.connect(Path(tempfile.mkdtemp()) / "t.db")


def _seed(con):
    """Два макета: A прогнан дважды (второй раз аспект 1 стал чистым),
    B — один раз. Фидбек — во всех четырёх формах note_type."""
    c1 = S.record_check(con, _report("a" * 64, {1: D.STATUS_VIOLATION, 2: D.STATUS_MANUAL}),
                        "/Users/private/reports/a_1.md")
    c2 = S.record_check(con, _report("a" * 64, {1: D.STATUS_COMPLIANT, 2: D.STATUS_MANUAL},
                                     tokens=CACHED_TOKENS), "/Users/private/reports/a_2.md")
    c3 = S.record_check(con, _report("b" * 64, {1: D.STATUS_VIOLATION, 2: D.STATUS_COMPLIANT},
                                     name="Другой макет.pdf"), "")
    S.record_feedback(con, c1, 1, "Наименование", D.STATUS_VIOLATION, "up", "", None)
    S.record_feedback(con, c1, 2, "Состав", D.STATUS_MANUAL, "down",
                      "текст макета дословно", "designer")
    con.execute("INSERT INTO feedback (ts, check_id, aspect_id, aspect_name, "
                "verdict_status, rating, note, note_type) VALUES "
                "('2026-08-31 10:00:00', ?, 21, 'Отличительные признаки (клеймы)', "
                "'соответствует', 'up', 'секретная пометка', 'замечание дизайнеру')", (c1,))
    con.execute("INSERT INTO feedback (ts, check_id, aspect_id, aspect_name, "
                "verdict_status, rating, note, note_type) VALUES "
                "('2026-08-31 10:01:00', ?, 6, 'Пищевая ценность', "
                "'возможное нарушение', NULL, '', 'на перепрогон')", (c2,))
    S.record_feedback(con, c3, 21, "Заявленные особенности", D.STATUS_COMPLIANT,
                      None, "", "supplier")
    con.commit()
    return c1, c2, c3


def test_note_type_four_forms_normalize():
    """NULL / 'none' / старые русские подписи / ключи → четыре ключа."""
    assert D.normalize_note_type(None) == "none"
    assert D.normalize_note_type("") == "none"
    assert D.normalize_note_type("none") == "none"
    assert D.normalize_note_type("designer") == "designer"
    assert D.normalize_note_type("замечание дизайнеру") == "designer"
    assert D.normalize_note_type("на перепрогон") == "manual"
    assert D.normalize_note_type("прочее") == "none"
    assert D.normalize_note_type("что-то новое") == "none"
    assert all(D.normalize_note_type(k) == k for k in S.NOTE_TYPES)


def test_price_longest_prefix_and_unknown_model():
    """gpt-5.4-mini не должен попадать под цену gpt-5.4; неизвестная модель
    даёт None, а не заниженную сумму."""
    prices = CFG["dashboard"]["prices_usd_per_1m"]
    assert D.price_for("gpt-5.4-mini-2026-03-17", prices) == (0.75, 4.5)
    assert D.price_for("gpt-5.4-2026-03-05", prices) == (2.5, 15.0)
    assert D.price_for("claude-x", prices) is None
    cost, unknown = D.cost_of(FULL_TOKENS, prices)
    assert unknown == [] and abs(cost - (200_000 * 2.5 + 30_000 * 15 + 1_000 * .75 + 100 * 4.5) / 1e6) < 1e-6
    cost, unknown = D.cost_of({"claude-x": {"prompt": 10, "completion": 1}}, prices)
    assert cost is None and unknown == ["claude-x"]
    assert D.cost_of({}, prices) == (0.0, [])


def test_main_calls_alias_and_fallback():
    """MAIN_MODEL из .env — короткий алиас; без него — модель с наибольшим
    расходом prompt-токенов."""
    assert D.main_calls(FULL_TOKENS, "gpt-5.4") == 33  # алиас покрывает и mini
    assert D.main_calls(FULL_TOKENS, "gpt-5.4-2026") == 32
    assert D.main_calls(FULL_TOKENS, None) == 32
    assert D.main_calls(CACHED_TOKENS, "gpt-5.4-2026") == 1
    assert D.main_calls({}, None) == 0


def test_load_checks_flags_cached_runs_and_tolerates_bad_json():
    con = _tmp_con()
    _seed(con)
    con.execute("UPDATE checks SET categories='не json', tokens_json=NULL WHERE id=3")
    con.commit()
    checks = D.load_checks(con, CFG, "gpt-5.4-2026")
    assert [c["id"] for c in checks] == [1, 2, 3]
    assert checks[0]["full_run"] and not checks[1]["full_run"]
    assert checks[0]["cost_usd"] > checks[1]["cost_usd"] > 0
    assert checks[0]["layout"] == "Секретный макет" and checks[0]["minutes"] == 5.0
    assert checks[0]["when"] is not None
    assert checks[2]["categories_list"] == [] and checks[2]["tokens"] == {}
    assert checks[2]["cost_usd"] == 0.0 and not checks[2]["full_run"]


def test_record_check_writes_verdicts_without_texts():
    """record_check пишет строку на аспект: статус, применимость, число
    цитат; текстов объяснений в таблице нет."""
    con = _tmp_con()
    cid = S.record_check(con, _report("c" * 64, {1: D.STATUS_VIOLATION}))
    rows = S.fetch_verdicts(con)
    assert [(r["check_id"], r["aspect_id"], r["status"], r["applicable"], r["n_citations"])
            for r in rows] == [(cid, 1, D.STATUS_VIOLATION, 1, 1),
                               (cid, 11, D.STATUS_COMPLIANT, 0, 0)]
    cols = {r[1] for r in con.execute("PRAGMA table_info(verdicts)")}
    assert "explanation" not in cols and "text" not in cols


def test_backfill_verdicts_from_reports_is_idempotent():
    """Старые прогоны без verdicts досыпаются из отчёта по ИМЕНИ файла
    (путь в базе — чужой абсолютный); повтор ничего не дублирует; прогон
    без отчёта — в missing."""
    tmp = Path(tempfile.mkdtemp())
    con = S.connect(tmp / "t.db")
    rep = _report("d" * 64, {1: D.STATUS_MANUAL, 2: D.STATUS_COMPLIANT})
    (tmp / "old_1.json").write_text(json.dumps(rep, ensure_ascii=False), encoding="utf-8")
    con.execute("INSERT INTO checks (ts, source_pdf, report_path) VALUES "
                "('2026-08-31 12:00:00', 'x.pdf', '/Users/other/mac/reports/old_1.md')")
    con.execute("INSERT INTO checks (ts, source_pdf, report_path) VALUES "
                "('2026-08-31 12:05:00', 'y.pdf', '/Users/other/mac/reports/gone.md')")
    con.commit()
    res = S.backfill_verdicts(con, tmp)
    assert res == {"filled": [1], "missing": [2]}
    assert len(S.fetch_verdicts(con)) == 3
    res2 = S.backfill_verdicts(con, tmp)
    assert res2 == {"filled": [], "missing": [2]} and len(S.fetch_verdicts(con)) == 3


def test_aspect_stats_latest_run_per_layout():
    """Доля проблемных считается по ПОСЛЕДНЕМУ прогону каждого макета: A
    прогнан дважды — берётся второй (аспект 1 чистый); неприменимый — в na."""
    con = _tmp_con()
    _seed(con)
    names = {1: "Наименование продукции", 2: "Состав", 11: "Импортёр"}
    stats = {a["aspect_id"]: a for a in D.aspect_status_stats(con, names)}
    assert D.latest_check_ids(con) == [2, 3]
    assert (stats[1]["violation"], stats[1]["ok"], stats[1]["applicable"]) == (1, 1, 2)
    assert stats[1]["problem_share"] == 0.5
    assert stats[2]["manual"] == 1 and stats[2]["problem_share"] == 0.5
    assert stats[11]["na"] == 2 and stats[11]["applicable"] == 0
    all_runs = {a["aspect_id"]: a for a in D.aspect_status_stats(con, names, latest_only=False)}
    assert all_runs[1]["violation"] == 2
    assert D.aspect_status_stats(_tmp_con(), names) == []  # пустая база не падает


def test_feedback_names_from_yaml_and_ratings_skip_null():
    """Аспект 21 в базе под двумя именами — в статистике одно, из
    aspects.yaml; запись без оценки не входит в 👍/👎."""
    con = _tmp_con()
    _seed(con)
    names = D.aspect_names()
    fb = D.load_feedback(con, names)
    assert {f["aspect"] for f in fb if f["aspect_id"] == 21} == {"Заявленные особенности"}
    assert [f["decision"] for f in fb] == ["none", "designer", "designer", "manual", "supplier"]
    ratings = {r["aspect_id"]: r for r in D.rating_stats(fb, names)}
    assert 6 not in ratings                        # rating NULL — не оценка
    assert ratings[21]["rated"] == 1 and ratings[21]["agreement"] == 1.0
    assert ratings[2]["down"] == 1 and ratings[2]["agreement"] == 0.0
    assert D.rating_stats([], names) == []


def test_decision_stats_counts_legacy_forms():
    con = _tmp_con()
    _seed(con)
    fb = D.load_feedback(con, {})
    counts = {d["decision"]: d["count"] for d in D.decision_stats(fb)}
    assert counts == {"none": 1, "designer": 2, "supplier": 1, "manual": 1}
    assert [d["decision"] for d in D.decision_stats([])] == list(S.NOTE_TYPES)


def test_layout_vision_stats_and_table_fallback():
    layout = {"text_layer_coverage": 0.8, "unread_layer_words": ["а", "б"], "missing": ["x"],
              "regions": [
                  {"id": "r1", "status": "прочитано", "human_edited": True},
                  {"id": "r2", "status": "требует ручной проверки", "invented_words": ["w"]},
                  {"id": "r3", "status": "прочитано", "human_confirmed": True}]}
    st = D.layout_vision_stats(layout, "L")
    assert st == {"layout": "L", "regions": 3, "manual": 1, "edited": 1, "confirmed": 1,
                  "invented": 1, "coverage": 0.8, "unread_words": 2, "missing": 1}
    assert D.layout_vision_stats({}, "пусто")["regions"] == 0
    # живой layout по имени PDF из журнала
    tmp = Path(tempfile.mkdtemp())
    (tmp / "Секретный макет.json").write_text(json.dumps(layout), encoding="utf-8")
    con = _tmp_con()
    _seed(con)
    checks = D.load_checks(con, CFG)
    live = D.vision_stats(con, checks, tmp)
    assert [v["layout"] for v in live] == ["Секретный макет"]   # один раз, не дважды
    # нет layout'ов и нет таблицы → пусто; есть таблица → из неё
    assert D.vision_stats(con, checks, tmp / "нет") == []
    D.save_vision_stats(con, [st])
    assert D.vision_stats(con, checks, None) == [st]


def test_pick_db_prefers_nonempty_work_db():
    tmp = Path(tempfile.mkdtemp())
    cfg = {"ui": {"db": str(tmp / "work.db")},
           "dashboard": {"demo_db": str(tmp / "demo.db")}}
    # абсолютные пути в конфиге перекрывают ROOT (Path('/a') / '/b' == '/b')
    assert D.pick_db(cfg) == (tmp / "work.db", False)   # ничего нет → рабочая (пустая)
    S.connect(tmp / "demo.db").close()
    assert D.pick_db(cfg) == (tmp / "demo.db", True)    # пустая рабочая → демо
    con = S.connect(tmp / "work.db")
    S.record_check(con, _report("e" * 64, {1: D.STATUS_COMPLIANT}))
    con.close()
    assert D.pick_db(cfg) == (tmp / "work.db", False)   # есть прогон → рабочая


def test_load_all_summary():
    con = _tmp_con()
    _seed(con)
    data = D.load_all(con, CFG, None, "gpt-5.4-2026")
    s = data["summary"]
    assert s["n_checks"] == 3 and s["n_layouts"] == 2 and s["n_full_runs"] == 2
    assert s["n_feedback"] == 5 and s["n_rated"] == 3 and s["agreement"] == 0.667
    assert s["cost_total_usd"] > 0 and s["cost_unknown_runs"] == 0
    assert data["vision"] == [] and len(data["decisions"]) == 4


def test_make_demo_db_strips_private_data():
    """Демо-копия: имена макетов → «Макет A/B», пути пусты, заметки скрыты,
    вердикты и метрики зрения перенесены; в БАЙТАХ файла нет исходных
    имён и путей (сборка с нуля + VACUUM); повторная сборка идемпотентна."""
    from evaluation.make_demo_db import build_demo
    tmp = Path(tempfile.mkdtemp())
    con = S.connect(tmp / "work.db")
    _seed(con)
    con.close()
    layouts = tmp / "layouts"
    layouts.mkdir()
    (layouts / "Секретный макет.json").write_text(json.dumps(
        {"text_layer_coverage": 0.9, "regions": [{"id": "r1", "status": "прочитано",
                                                  "text": "Секретный состав"}]}),
        encoding="utf-8")
    for _ in range(2):
        res = build_demo(tmp / "work.db", tmp / "demo.db", layouts)
    assert res["checks"] == 3 and res["verdicts"] == 9 and res["feedback"] == 5
    assert res["layouts"] == 1
    raw = (tmp / "demo.db").read_bytes()
    for needle in ("Секретный", "/Users", "текст макета дословно", "Другой макет", "секретная пометка"):
        assert needle.encode("utf-8") not in raw, needle
    d = sqlite3.connect(tmp / "demo.db")
    d.row_factory = sqlite3.Row
    checks = [dict(r) for r in d.execute("SELECT * FROM checks ORDER BY id")]
    assert [c["source_pdf"] for c in checks] == ["Макет A.pdf", "Макет A.pdf", "Макет B.pdf"]
    assert checks[0]["source_sha256"] == checks[1]["source_sha256"] != checks[2]["source_sha256"]
    assert all(c["report_path"] == "" for c in checks)
    assert checks[0]["tokens_json"] and json.loads(checks[0]["tokens_json"])  # токены целы
    notes = {r[0] for r in d.execute("SELECT note FROM feedback")}
    assert notes == {"", "[заметка скрыта в демо-копии]"}
    assert d.execute("SELECT COUNT(*) FROM verdicts").fetchone()[0] == 9
    assert [tuple(r)[:2] for r in d.execute("SELECT layout, regions FROM vision_stats")] == [("Макет A", 1)]
    # дашборд читает демо так же, как рабочую базу
    data = D.load_all(S.connect(tmp / "demo.db"), CFG, tmp / "нет")
    assert data["summary"]["n_checks"] == 3 and data["vision"][0]["layout"] == "Макет A"


def test_committed_demo_db_has_no_private_strings():
    """Снимок в репозитории: ни путей, ни имён файлов макетов, ни заметок;
    дополнительно — список запрещённых подстрок из data/demo_forbidden.txt
    (личный, не в git), если он есть на этой машине."""
    demo = ROOT / CFG["dashboard"]["demo_db"]
    if not demo.exists():
        return  # чистый клон без снимка — проверять нечего
    con = sqlite3.connect(demo)
    for (name,) in con.execute("SELECT DISTINCT source_pdf FROM checks"):
        assert re.fullmatch(r"Макет [A-Z]+\.pdf", name or ""), name
    for (p,) in con.execute("SELECT DISTINCT report_path FROM checks"):
        assert not p, p
    for (note,) in con.execute("SELECT DISTINCT note FROM feedback"):
        assert note in ("", "[заметка скрыта в демо-копии]"), note[:40]
    raw = demo.read_bytes()
    for needle in (b"/Users/", b"/home/", b"C:\\"):
        assert needle not in raw, needle
    forbidden = ROOT / "data" / "demo_forbidden.txt"
    if forbidden.exists():
        for line in forbidden.read_text(encoding="utf-8").splitlines():
            word = line.strip()
            if word and not word.startswith("#"):
                assert word.encode("utf-8").lower() not in raw.lower(), word


def test_step4_renders_all_charts_headless():
    """Смоук шага «4 · Мониторинг» в AppTest: на демо-копии из синтетической
    базы рисуются все семь графиков и KPI, исключений нет; подмена базы —
    через dashboard.pick_db (рабочая база машины не трогается)."""
    from streamlit.testing.v1 import AppTest
    from evaluation.make_demo_db import build_demo
    tmp = Path(tempfile.mkdtemp())
    con = S.connect(tmp / "work.db")
    _seed(con)
    con.close()
    build_demo(tmp / "work.db", tmp / "demo.db", None)
    real_pick = D.pick_db
    D.pick_db = lambda cfg: (tmp / "demo.db", True)
    try:
        at = AppTest.from_file(str(ROOT / "labelcheck" / "app.py"), default_timeout=90)
        at.run()
        at.session_state["nav"] = "4 · Мониторинг"
        at.run()
        assert not at.exception, at.exception
        page = " ".join(m.value for m in at.markdown)
        for needle in ("1. Статусы вердиктов", "2. Проблемные аспекты",
                       "3. Согласие эксперта", "4. Кому уходят замечания",
                       "5. Стоимость прогонов", "6. Когда проверяли",
                       "7. Качество распознавания"):
            assert needle in page, needle
        infos = " ".join(i.value for i in at.info)
        assert "демо-копия" in infos
        labels = [m.label for m in at.metric]
        assert labels == ["Проверок", "Макетов", "Оценок эксперта",
                          "Согласие с системой", "Стоимость, $"], labels
        values = [m.value for m in at.metric]
        assert values[0] == "3" and values[1] == "2" and values[3] == "67%", values
    finally:
        D.pick_db = real_pick


def test_step4_empty_db_shows_hint():
    """Пустой журнал — подсказка вместо графиков, без исключений."""
    from streamlit.testing.v1 import AppTest
    tmp = Path(tempfile.mkdtemp())
    S.connect(tmp / "empty.db").close()
    real_pick = D.pick_db
    D.pick_db = lambda cfg: (tmp / "empty.db", False)
    try:
        at = AppTest.from_file(str(ROOT / "labelcheck" / "app.py"), default_timeout=90)
        at.run()
        at.session_state["nav"] = "4 · Мониторинг"
        at.run()
        assert not at.exception, at.exception
        assert any("Проверок ещё не было" in i.value for i in at.info)
        assert not at.metric
    finally:
        D.pick_db = real_pick


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
