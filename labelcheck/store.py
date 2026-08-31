"""Хранилище UI (блок C, день 9): SQLite-журнал проверок и фидбека +
правки layout'а человеком.

Зачем отдельный модуль без Streamlit: слой данных тестируется без API и
без UI (tests/test_store.py), а схема SQLite — общая точка для дашборда
мониторинга Дня 10 (рубрика: 👍/👎 + комментарий + 5+ графиков).

Таблицы:
- checks   — журнал прогонов вердиктов: макет, категории, счётчики
             статусов, расход токенов, путь к отчёту;
- feedback — оценка вердикта человеком (up/down) и заметка с типом
             («на перепрогон» / «замечание дизайнеру» / «прочее»).

Правки регионов человеком (ТЗ §2 п.4 — «пользователь может поправить
распознанное ДО анализа») пишутся в сам layout-JSON: текст заменяется,
история правок остаётся в поле edits региона, при первом сохранении
рядом кладётся нетронутая копия <имя>.orig.json.
"""

import json
import sqlite3
import time
from pathlib import Path

from labelcheck.retrieval import ROOT, load_config

NOTE_TYPES = ("на перепрогон", "замечание дизайнеру", "прочее")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS checks (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  ts            TEXT NOT NULL,
  source_pdf    TEXT,
  source_sha256 TEXT,
  categories    TEXT,   -- JSON-список категорий
  category_scan TEXT,   -- targeted / full / manual
  n_violation   INTEGER,
  n_manual      INTEGER,
  n_ok          INTEGER,
  n_na          INTEGER,
  tokens_json   TEXT,   -- расход по моделям (JSON)
  seconds       REAL,
  report_path   TEXT
);
CREATE TABLE IF NOT EXISTS feedback (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  ts             TEXT NOT NULL,
  check_id       INTEGER REFERENCES checks(id),
  aspect_id      INTEGER,
  aspect_name    TEXT,
  verdict_status TEXT,
  rating         TEXT,   -- 'up' / 'down' / NULL (только заметка)
  note           TEXT,
  note_type      TEXT    -- одно из NOTE_TYPES / NULL
);
"""


def db_path(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return ROOT / cfg["ui"]["db"]


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    """Соединение с базой; схема создаётся идемпотентно."""
    p = Path(path) if path else db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.row_factory = sqlite3.Row
    con.executescript(_SCHEMA)
    return con


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ── журнал проверок ──────────────────────────────────────────────────────────

def record_check(con: sqlite3.Connection, report: dict,
                 report_path: str = "") -> int:
    """Прогон вердиктов → строка в checks. Возвращает id записи."""
    m = report["meta"]
    statuses = {"возможное нарушение": 0, "требует ручной проверки": 0,
                "соответствует": 0}
    n_na = 0
    for v in report["verdicts"]:
        if not v["applicable"]:
            n_na += 1
        else:
            statuses[v["status"]] += 1
    cur = con.execute(
        "INSERT INTO checks (ts, source_pdf, source_sha256, categories, "
        "category_scan, n_violation, n_manual, n_ok, n_na, tokens_json, "
        "seconds, report_path) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (_now(), m.get("source_pdf"), m.get("source_sha256"),
         json.dumps(sorted(m.get("categories", {})), ensure_ascii=False),
         m.get("category_scan"),
         statuses["возможное нарушение"], statuses["требует ручной проверки"],
         statuses["соответствует"], n_na,
         json.dumps(m.get("tokens", {}), ensure_ascii=False),
         m.get("seconds"), str(report_path)))
    con.commit()
    return cur.lastrowid


def fetch_checks(con: sqlite3.Connection, limit: int = 50) -> list[dict]:
    rows = con.execute("SELECT * FROM checks ORDER BY id DESC LIMIT ?",
                       (limit,)).fetchall()
    return [dict(r) for r in rows]


# ── фидбек ───────────────────────────────────────────────────────────────────

def record_feedback(con: sqlite3.Connection, check_id: int, aspect_id: int,
                    aspect_name: str, verdict_status: str,
                    rating: str | None = None, note: str = "",
                    note_type: str | None = None) -> int:
    """Оценка/заметка по одному вердикту. rating — 'up'/'down'/None."""
    if rating not in ("up", "down", None):
        raise ValueError(f"rating: ожидаю up/down/None, получено {rating!r}")
    if note_type is not None and note_type not in NOTE_TYPES:
        raise ValueError(f"note_type: ожидаю {NOTE_TYPES}, получено {note_type!r}")
    cur = con.execute(
        "INSERT INTO feedback (ts, check_id, aspect_id, aspect_name, "
        "verdict_status, rating, note, note_type) VALUES (?,?,?,?,?,?,?,?)",
        (_now(), check_id, aspect_id, aspect_name, verdict_status,
         rating, note.strip(), note_type))
    con.commit()
    return cur.lastrowid


def fetch_feedback(con: sqlite3.Connection,
                   check_id: int | None = None) -> list[dict]:
    if check_id is None:
        rows = con.execute("SELECT * FROM feedback ORDER BY id").fetchall()
    else:
        rows = con.execute("SELECT * FROM feedback WHERE check_id=? ORDER BY id",
                           (check_id,)).fetchall()
    return [dict(r) for r in rows]


def export_notes(con: sqlite3.Connection,
                 check_id: int | None = None) -> list[dict]:
    """Заметки с типом (для выгрузки «на перепрогон» / «дизайнеру»)."""
    return [f for f in fetch_feedback(con, check_id)
            if (f.get("note") or "").strip()]


# ── правки layout'а человеком ────────────────────────────────────────────────

def apply_region_edit(layout: dict, region_id: str, new_text: str,
                      editor: str = "human") -> bool:
    """Правка текста региона: заменяет text, историю кладёт в region.edits,
    статус становится «прочитано» (человек — истина в последней инстанции).
    Возвращает False, если текст не изменился; KeyError — нет региона."""
    for region in layout.get("regions", []):
        if region["id"] != region_id:
            continue
        old = region.get("text") or ""
        if old == new_text:
            return False
        region.setdefault("edits", []).append(
            {"ts": _now(), "editor": editor, "was": old})
        region["text"] = new_text
        region["status"] = "прочитано"
        region["status_reason"] = "текст выверен человеком"
        region["human_edited"] = True
        return True
    raise KeyError(f"регион {region_id!r} не найден в layout")


def save_layout(layout: dict, path: str | Path) -> Path:
    """Сохраняет layout-JSON; при первом сохранении рядом остаётся
    нетронутая копия <имя>.orig.json (канонический layout не теряется)."""
    p = Path(path)
    orig = p.with_suffix(".orig.json")
    if p.exists() and not orig.exists():
        orig.write_bytes(p.read_bytes())
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(layout, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    tmp.replace(p)  # атомарно, как кэши rewrite/verdict
    return p
