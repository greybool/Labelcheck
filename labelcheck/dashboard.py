"""Мониторинг (День 10): подготовка данных для дашборда БЕЗ Streamlit.

Зачем отдельный модуль: цифры для графиков считаются здесь на чистом
Python (списки словарей) и проверяются тестами без интерфейса
(tests/test_dashboard.py), а labelcheck/dashboard_ui.py только рисует.
Источники: SQLite-журнал (labelcheck/store.py: checks, verdicts, feedback)
и layout-JSON макетов (data/layouts/) — для качества распознавания.

Главная работа модуля — нормализация «грязных» данных, накопленных за
время приёмки (handoff-09 §4):
- feedback.note_type встречается в четырёх видах: NULL, 'none', русские
  слова старой схемы («замечание дизайнеру», «на перепрогон», «прочее»)
  и ключи designer/supplier/manual — сводится к четырём ключам;
- feedback.aspect_name менялся у аспекта 21 — группировка только по
  aspect_id, имена берутся из aspects.yaml;
- tokens_json у повторных прогонов из UI искажён кэшем вердиктов
  (только план работ, ~1,4k токенов) — такие прогоны помечаются «из кэша»;
- checks.ts — локальное время строкой; checks.categories — JSON-список.

CLI (без UI): python -m labelcheck.dashboard            — сводка текстом
              python -m labelcheck.dashboard --backfill — досыпать verdicts
              для старых прогонов из отчётов data/reports/ui/*.json
"""

import json
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from labelcheck.aspects import load_aspects
from labelcheck.retrieval import ROOT, load_config
from labelcheck.store import NOTE_TYPES, backfill_verdicts, connect, db_path

STATUS_VIOLATION = "возможное нарушение"
STATUS_MANUAL = "требует ручной проверки"
STATUS_COMPLIANT = "соответствует"
STATUSES = (STATUS_VIOLATION, STATUS_MANUAL, STATUS_COMPLIANT)
REGION_OK = "прочитано"

# Старая схема заметок (первые прогоны 31.08–01.09) → ключи NOTE_TYPES.
# «на перепрогон» = человек хотел проверить систему ещё раз — по смыслу
# ближе всего к «проверить самому».
NOTE_TYPE_LEGACY = {
    "замечание дизайнеру": "designer",
    "на перепрогон": "manual",
    "прочее": "none",
}
NOTE_TYPE_LABELS = {"none": "Ничего не требуется", "designer": "Дизайнеру",
                    "supplier": "Поставщику", "manual": "Проверить самому"}


def normalize_note_type(value) -> str:
    """NULL / '' / старые русские подписи / ключи → один из NOTE_TYPES.
    Неизвестное значение считается «ничего не требуется» (а не роняет
    дашборд): это журнал, а не источник вердиктов."""
    if value is None:
        return "none"
    v = str(value).strip().lower()
    if v in NOTE_TYPES:
        return v
    return NOTE_TYPE_LEGACY.get(v, "none")


def aspect_names(aspects: dict | None = None) -> dict[int, str]:
    """{aspect_id: имя} из aspects.yaml — единственный источник имён."""
    aspects = aspects or load_aspects()
    return {int(a["id"]): a["name"] for a in aspects["aspects"]}


# ── стоимость ────────────────────────────────────────────────────────────────

def price_for(model: str, prices: dict) -> tuple[float, float] | None:
    """Цена [вход, выход] за 1M токенов по самому длинному совпавшему
    префиксу имени модели (gpt-5.4-mini раньше gpt-5.4). None — модель
    в прайсе не описана."""
    best = None
    for prefix, pair in prices.items():
        if model.startswith(prefix) and (best is None or len(prefix) > len(best[0])):
            best = (prefix, pair)
    return (float(best[1][0]), float(best[1][1])) if best else None


def cost_of(tokens: dict, prices: dict) -> tuple[float | None, list[str]]:
    """(стоимость USD, [модели без цены]). Если хоть одна модель без
    цены — стоимость None: лучше «неизвестно», чем заниженная сумма."""
    total, unknown = 0.0, []
    for model, t in (tokens or {}).items():
        pair = price_for(model, prices)
        if pair is None:
            unknown.append(model)
            continue
        total += (t.get("prompt", 0) * pair[0] + t.get("completion", 0) * pair[1]) / 1e6
    return (None if unknown else round(total, 4)), unknown


def main_calls(tokens: dict, main_model_prefix: str | None) -> int:
    """Число вызовов MAIN-модели в прогоне. Имя из .env (MAIN_MODEL) может
    быть коротким алиасом («gpt-5.4»), в tokens_json — полное; без известного
    алиаса берём модель с наибольшим расходом prompt-токенов."""
    if not tokens:
        return 0
    if main_model_prefix:
        hits = [t for m, t in tokens.items() if m.startswith(main_model_prefix)]
        if hits:
            return sum(t.get("calls", 0) for t in hits)
    biggest = max(tokens.values(), key=lambda t: t.get("prompt", 0))
    return biggest.get("calls", 0)


# ── журнал проверок ──────────────────────────────────────────────────────────

def _parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s[:19], fmt)
        except ValueError:
            continue
    return None


def layout_label(source_pdf: str | None) -> str:
    """Имя макета для подписей: имя файла без расширения и папок."""
    return Path(source_pdf).stem if source_pdf else "—"


def load_checks(con: sqlite3.Connection, cfg: dict,
                main_model_prefix: str | None = None) -> list[dict]:
    """Все прогоны с распакованными полями и оценкой стоимости."""
    dcfg = cfg["dashboard"]
    prices = dcfg.get("prices_usd_per_1m", {})
    min_calls = int(dcfg.get("full_run_min_main_calls", 20))
    rows = con.execute("SELECT * FROM checks ORDER BY id").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        tokens = json.loads(d.get("tokens_json") or "{}")
        try:
            cats = json.loads(d.get("categories") or "[]")
        except json.JSONDecodeError:
            cats = []
        cost, unknown = cost_of(tokens, prices)
        calls = main_calls(tokens, main_model_prefix)
        d.update({
            "tokens": tokens,
            "categories_list": cats if isinstance(cats, list) else [],
            "layout": layout_label(d.get("source_pdf")),
            "when": _parse_ts(d.get("ts")),
            "tokens_total": sum(t.get("prompt", 0) + t.get("completion", 0)
                                for t in tokens.values()),
            "main_calls": calls,
            "full_run": calls >= min_calls,
            "cost_usd": cost,
            "models_without_price": unknown,
            "minutes": round((d.get("seconds") or 0) / 60, 1),
        })
        out.append(d)
    return out


# ── вердикты по аспектам ─────────────────────────────────────────────────────

def latest_check_ids(con: sqlite3.Connection) -> list[int]:
    """Последний прогон по каждому макету (ключ — SHA256 файла, без него —
    имя). Один макет mango прогнан девять раз с одинаковыми вердиктами из
    кэша — считать его девять раз значит исказить картину по аспектам."""
    rows = con.execute(
        "SELECT MAX(id) FROM checks GROUP BY COALESCE(source_sha256, source_pdf)"
    ).fetchall()
    return sorted(r[0] for r in rows)


def aspect_status_stats(con: sqlite3.Connection, names: dict[int, str],
                        latest_only: bool = True) -> list[dict]:
    """По каждому аспекту: сколько раз он был нарушением / ручной / чистым
    среди ПРИМЕНИМЫХ вердиктов, сколько раз неприменим, и доля проблемных.
    Источник — таблица verdicts (после backfill покрывает все прогоны).
    latest_only — только последний прогон каждого макета (актуальное
    состояние макетов, без повторов из кэша)."""
    if latest_only:
        ids = latest_check_ids(con)
        marks = ",".join("?" * len(ids)) or "NULL"
        rows = con.execute(
            f"SELECT aspect_id, status, applicable FROM verdicts "
            f"WHERE check_id IN ({marks})", ids).fetchall()
    else:
        rows = con.execute("SELECT aspect_id, status, applicable FROM verdicts").fetchall()
    acc: dict[int, Counter] = defaultdict(Counter)
    for r in rows:
        aid = int(r["aspect_id"])
        if not r["applicable"]:
            acc[aid]["na"] += 1
        elif r["status"] in STATUSES:
            acc[aid][r["status"]] += 1
    out = []
    for aid, c in acc.items():
        applicable = sum(c[s] for s in STATUSES)
        problems = c[STATUS_VIOLATION] + c[STATUS_MANUAL]
        out.append({
            "aspect_id": aid, "aspect": names.get(aid, f"аспект {aid}"),
            "violation": c[STATUS_VIOLATION], "manual": c[STATUS_MANUAL],
            "ok": c[STATUS_COMPLIANT], "na": c["na"], "applicable": applicable,
            "problem_share": round(problems / applicable, 3) if applicable else 0.0,
        })
    out.sort(key=lambda x: (-x["problem_share"], x["aspect_id"]))
    return out


# ── оценки человека ──────────────────────────────────────────────────────────

def load_feedback(con: sqlite3.Connection, names: dict[int, str]) -> list[dict]:
    """Все записи фидбека с нормализованным note_type и именем аспекта из
    aspects.yaml (aspect_name в базе у аспекта 21 менялся)."""
    rows = con.execute("SELECT * FROM feedback ORDER BY id").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        aid = int(d["aspect_id"])
        d["aspect"] = names.get(aid, d.get("aspect_name") or f"аспект {aid}")
        d["decision"] = normalize_note_type(d.get("note_type"))
        d["has_note"] = bool((d.get("note") or "").strip())
        out.append(d)
    return out


def rating_stats(feedback: list[dict], names: dict[int, str]) -> list[dict]:
    """👍/👎 по аспектам: согласие эксперта с системой. Записи без оценки
    (только адресат) не считаются."""
    acc: dict[int, Counter] = defaultdict(Counter)
    for f in feedback:
        if f.get("rating") in ("up", "down"):
            acc[int(f["aspect_id"])][f["rating"]] += 1
    out = []
    for aid, c in acc.items():
        total = c["up"] + c["down"]
        out.append({"aspect_id": aid, "aspect": names.get(aid, f"аспект {aid}"),
                    "up": c["up"], "down": c["down"], "rated": total,
                    "agreement": round(c["up"] / total, 3) if total else None})
    out.sort(key=lambda x: (x["agreement"] if x["agreement"] is not None else 2,
                            x["aspect_id"]))
    return out


def decision_stats(feedback: list[dict]) -> list[dict]:
    """Адресаты решений: сколько замечаний ушло дизайнеру / поставщику /
    на ручную проверку / снято. Порядок — как в NOTE_TYPES."""
    c = Counter(f["decision"] for f in feedback)
    return [{"decision": k, "label": NOTE_TYPE_LABELS[k], "count": c.get(k, 0)}
            for k in NOTE_TYPES]


# ── качество распознавания (layout-JSON) ─────────────────────────────────────

def layout_vision_stats(layout: dict, label: str) -> dict:
    """Метрики одного layout'а: блоков всего / сомнительных / правлено /
    подтверждено человеком, покрытие текстового слоя, блоков с
    «выдумками». Числа, без текстов — годятся для демо-БД.

    Четыре группы блоков не пересекаются (для стопки на графике): правка и
    подтверждение переводят блок в «прочитано», а подтверждённый и потом
    правленный блок считается правленным."""
    regions = layout.get("regions", [])
    return {
        "layout": label,
        "regions": len(regions),
        "manual": sum(1 for r in regions if r.get("status") != REGION_OK),
        "edited": sum(1 for r in regions if r.get("human_edited")),
        "confirmed": sum(1 for r in regions
                         if r.get("human_confirmed") and not r.get("human_edited")),
        "invented": sum(1 for r in regions if r.get("invented_words")),
        "coverage": layout.get("text_layer_coverage"),
        "unread_words": len(layout.get("unread_layer_words") or []),
        "missing": len(layout.get("missing") or []),
    }


def vision_stats(con: sqlite3.Connection, checks: list[dict],
                 layouts_dir: Path | None) -> list[dict]:
    """Качество распознавания по макетам из журнала. Сначала — живые
    layout'ы (data/layouts/<имя PDF>.json: та копия, что соответствует
    проверкам из UI, с правками человека); если ни одного нет (чистый клон,
    демо-БД) — таблица vision_stats, которую пишет make_demo_db."""
    out, seen = [], set()
    if layouts_dir:
        for c in checks:
            label = c["layout"]
            if label in seen or label == "—":
                continue
            path = Path(layouts_dir) / f"{label}.json"
            if not path.exists():
                continue
            seen.add(label)
            layout = json.loads(path.read_text(encoding="utf-8"))
            out.append(layout_vision_stats(layout, label))
    if out:
        return out
    has_table = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='vision_stats'"
    ).fetchone()
    if not has_table:
        return []
    return [dict(r) for r in con.execute("SELECT * FROM vision_stats ORDER BY layout")]


def save_vision_stats(con: sqlite3.Connection, rows: list[dict]) -> None:
    """Таблица-снимок для демо-БД (layout'ы в репозиторий не попадают)."""
    con.execute("DROP TABLE IF EXISTS vision_stats")
    con.execute("CREATE TABLE vision_stats (layout TEXT, regions INTEGER, "
                "manual INTEGER, edited INTEGER, confirmed INTEGER, "
                "invented INTEGER, coverage REAL, unread_words INTEGER, "
                "missing INTEGER)")
    con.executemany(
        "INSERT INTO vision_stats VALUES (?,?,?,?,?,?,?,?,?)",
        [(r["layout"], r["regions"], r["manual"], r["edited"], r["confirmed"],
          r["invented"], r["coverage"], r["unread_words"], r["missing"])
         for r in rows])
    con.commit()


# ── сборка ───────────────────────────────────────────────────────────────────

def pick_db(cfg: dict) -> tuple[Path, bool]:
    """(путь к базе, это демо?). Рабочая база — если существует и в ней есть
    хоть один прогон; иначе демо-копия из конфига; иначе рабочая (пустая)."""
    work = db_path(cfg)
    if work.exists():
        con = sqlite3.connect(work)
        try:
            n = con.execute("SELECT COUNT(*) FROM checks").fetchone()[0]
        except sqlite3.OperationalError:
            n = 0
        finally:
            con.close()
        if n:
            return work, False
    demo = ROOT / cfg["dashboard"]["demo_db"]
    if demo.exists():
        return demo, True
    return work, False


def load_all(con: sqlite3.Connection, cfg: dict | None = None,
             layouts_dir: Path | None = None,
             main_model_prefix: str | None = None) -> dict:
    """Всё для дашборда одним словарём (чистые списки словарей)."""
    cfg = cfg or load_config()
    names = aspect_names()
    checks = load_checks(con, cfg, main_model_prefix)
    feedback = load_feedback(con, names)
    ratings = rating_stats(feedback, names)
    rated = sum(r["rated"] for r in ratings)
    ups = sum(r["up"] for r in ratings)
    costs = [c["cost_usd"] for c in checks if c["cost_usd"] is not None]
    return {
        "checks": checks,
        "aspects": aspect_status_stats(con, names),
        "feedback": feedback,
        "ratings": ratings,
        "decisions": decision_stats(feedback),
        "vision": vision_stats(con, checks, layouts_dir),
        "summary": {
            "n_checks": len(checks),
            "n_layouts": len({c["source_sha256"] or c["layout"] for c in checks}),
            "n_full_runs": sum(1 for c in checks if c["full_run"]),
            "n_feedback": len(feedback),
            "n_rated": rated,
            "agreement": round(ups / rated, 3) if rated else None,
            "cost_total_usd": round(sum(costs), 2) if costs else None,
            "cost_unknown_runs": sum(1 for c in checks if c["cost_usd"] is None),
        },
    }


def _main(argv: list[str]) -> int:
    cfg = load_config()
    if "--backfill" in argv:
        con = connect()
        res = backfill_verdicts(con, ROOT / cfg["ui"]["reports_dir"])
        con.close()
        print(f"verdicts досыпаны для прогонов: {res['filled'] or '—'}; "
              f"отчёт не найден: {res['missing'] or '—'}")
        return 0
    path, is_demo = pick_db(cfg)
    con = connect(path)
    data = load_all(con, cfg, ROOT / "data" / "layouts")
    con.close()
    print(f"База: {path.name}{' (демо)' if is_demo else ''}")
    print("Сводка:", json.dumps(data["summary"], ensure_ascii=False))
    print("Проблемные аспекты (доля нарушение+ручная среди применимых):")
    for a in data["aspects"][:10]:
        print(f"  {a['aspect_id']:>2} {a['aspect'][:34]:<34} "
              f"{a['problem_share']:.0%}  (🔴{a['violation']} 🟡{a['manual']} 🟢{a['ok']})")
    print("Согласие эксперта по аспектам (👍 / оценено):")
    for r in data["ratings"]:
        print(f"  {r['aspect_id']:>2} {r['aspect'][:34]:<34} {r['up']}/{r['rated']}")
    print("Адресаты:", {d['label']: d['count'] for d in data["decisions"]})
    print("Распознавание:", data["vision"])
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
