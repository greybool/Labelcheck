"""Обезличенная демо-копия журнала для ревьюера (День 10).

Рабочая база data/labelcheck.db не в git (имена файлов макетов, пути на
машине владельца, заметки с текстами макетов). На чистом клоне дашборд был
бы пуст — поэтому в репозиторий кладётся демо-снимок data/labelcheck.demo.db,
собранный этим скриптом. Дашборд берёт рабочую базу, если она есть и не
пуста, иначе — демо (labelcheck/dashboard.py → pick_db).

Что вырезается (см. anonymize_* ниже):
- checks.source_pdf  → «Макет A.pdf», «Макет B.pdf»… (по порядку первого
  появления; один и тот же файл — одна буква, ключ — SHA256);
- checks.source_sha256 → SHA256 нового имени (связка прогонов сохраняется);
- checks.report_path → пусто (абсолютные пути);
- feedback.note → «[заметка скрыта в демо-копии]» у непустых (заметки могут
  содержать дословные тексты макетов); aspect_name остаётся — это имена
  аспектов из aspects.yaml;
- verdicts — переносятся как есть (статусы, без текстов);
- vision_stats — числа качества распознавания из data/layouts/ (сами
  layout'ы в git не попадают), под теми же обезличенными именами.

Снимок собирается заново, а не правится: в новый файл вставляются только
преобразованные строки, удалённых данных в свободных страницах SQLite нет.

Запуск (из корня, на машине с рабочей базой):
    python evaluation/make_demo_db.py            # data/labelcheck.demo.db
    python evaluation/make_demo_db.py --dst x.db # другой путь
Перед сборкой досыпать вердикты старым прогонам:
    python -m labelcheck.dashboard --backfill
Контроль: tests/test_dashboard.py (нет путей, имён файлов и заметок).
"""

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from labelcheck.dashboard import layout_vision_stats, save_vision_stats  # noqa: E402
from labelcheck.retrieval import load_config  # noqa: E402
from labelcheck.store import connect  # noqa: E402

NOTE_PLACEHOLDER = "[заметка скрыта в демо-копии]"
DEMO_NAME = "Макет {letter}.pdf"


def _letter(i: int) -> str:
    """0 → A, 25 → Z, 26 → AA …"""
    s = ""
    i += 1
    while i:
        i, rem = divmod(i - 1, 26)
        s = chr(65 + rem) + s
    return s


def build_name_map(checks: list[dict]) -> dict[str, str]:
    """{ключ макета (sha256 или имя): демо-имя} в порядке первого появления."""
    names: dict[str, str] = {}
    for c in checks:
        key = c.get("source_sha256") or c.get("source_pdf") or ""
        if key not in names:
            names[key] = DEMO_NAME.format(letter=_letter(len(names)))
    return names


def anonymize_check(c: dict, names: dict[str, str]) -> dict:
    key = c.get("source_sha256") or c.get("source_pdf") or ""
    demo = names[key]
    d = dict(c)
    d["source_pdf"] = demo
    d["source_sha256"] = hashlib.sha256(demo.encode("utf-8")).hexdigest()
    d["report_path"] = ""
    return d


def anonymize_feedback(f: dict) -> dict:
    d = dict(f)
    d["note"] = NOTE_PLACEHOLDER if (d.get("note") or "").strip() else ""
    return d


def build_demo(src: Path, dst: Path, layouts_dir: Path | None) -> dict:
    """Собирает демо-базу с нуля. Возвращает счётчики для вывода/тестов."""
    if dst.exists():
        dst.unlink()
    s = sqlite3.connect(src)
    s.row_factory = sqlite3.Row
    checks = [dict(r) for r in s.execute("SELECT * FROM checks ORDER BY id")]
    feedback = [dict(r) for r in s.execute("SELECT * FROM feedback ORDER BY id")]
    try:
        verdicts = [dict(r) for r in s.execute("SELECT * FROM verdicts ORDER BY id")]
    except sqlite3.OperationalError:  # база старее таблицы verdicts
        verdicts = []
    s.close()

    names = build_name_map(checks)
    d = connect(dst)  # схема — та же, что у рабочей базы
    for c in checks:
        a = anonymize_check(c, names)
        d.execute(
            "INSERT INTO checks (id, ts, source_pdf, source_sha256, categories, "
            "category_scan, n_violation, n_manual, n_ok, n_na, tokens_json, "
            "seconds, report_path) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (a["id"], a["ts"], a["source_pdf"], a["source_sha256"],
             a["categories"], a["category_scan"], a["n_violation"], a["n_manual"],
             a["n_ok"], a["n_na"], a["tokens_json"], a["seconds"], a["report_path"]))
    for v in verdicts:
        d.execute(
            "INSERT INTO verdicts (id, check_id, aspect_id, status, applicable, "
            "n_citations, downgraded, votes) VALUES (?,?,?,?,?,?,?,?)",
            (v["id"], v["check_id"], v["aspect_id"], v["status"], v["applicable"],
             v["n_citations"], v["downgraded"], v["votes"]))
    for f in feedback:
        a = anonymize_feedback(f)
        d.execute(
            "INSERT INTO feedback (id, ts, check_id, aspect_id, aspect_name, "
            "verdict_status, rating, note, note_type) VALUES (?,?,?,?,?,?,?,?,?)",
            (a["id"], a["ts"], a["check_id"], a["aspect_id"], a["aspect_name"],
             a["verdict_status"], a["rating"], a["note"], a["note_type"]))
    d.commit()

    vision_rows = []
    if layouts_dir:
        seen = set()
        for c in checks:
            key = c.get("source_sha256") or c.get("source_pdf") or ""
            stem = Path(c.get("source_pdf") or "").stem
            path = Path(layouts_dir) / f"{stem}.json"
            if key in seen or not stem or not path.exists():
                continue
            seen.add(key)
            layout = json.loads(path.read_text(encoding="utf-8"))
            vision_rows.append(layout_vision_stats(layout, Path(names[key]).stem))
    save_vision_stats(d, vision_rows)
    d.execute("VACUUM")
    d.close()
    return {"checks": len(checks), "verdicts": len(verdicts),
            "feedback": len(feedback), "layouts": len(vision_rows),
            "names": names}


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default=str(ROOT / cfg["ui"]["db"]))
    ap.add_argument("--dst", default=str(ROOT / cfg["dashboard"]["demo_db"]))
    ap.add_argument("--layouts", default=str(ROOT / "data" / "layouts"))
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    if not src.exists():
        print(f"нет рабочей базы: {src}")
        return 1
    res = build_demo(src, dst, Path(args.layouts))
    print(f"{dst}: прогонов {res['checks']}, вердиктов {res['verdicts']}, "
          f"оценок {res['feedback']}, макетов с метриками зрения {res['layouts']}")
    if res["verdicts"] == 0 and res["checks"]:
        print("ВНИМАНИЕ: таблица verdicts пуста — сначала "
              "python -m labelcheck.dashboard --backfill")
    return 0


if __name__ == "__main__":
    sys.exit(main())
