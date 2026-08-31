"""Стабильность вердиктов (День 8): совпадают ли статусы между прогонами.

Запуск из корня репозитория:
  python evaluation/stability.py data/reports/<run1>.json <run2>.json …

Вход — несколько JSON-отчётов вердикт-пайплайна по ОДНОМУ макету.
Выход — evaluation/metrics/stability.json (в git: только номера аспектов,
статусы и счётчики — ни одной строки текста макета) + таблица в stdout.

Смысл: LLM недетерминирована, пограничные аспекты меняют статус между
прогонами (в Дне 6 плавали аспекты 6 и 15). Метрика — доля аспектов с
одинаковым статусом во всех прогонах; нестабильные — кандидаты на
self-consistency / кэш вердиктов (BACKLOG).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from labelcheck.retrieval import load_config


def stability_table(reports: list[dict]) -> dict:
    """Отчёты одного макета → статусы по аспектам и доля стабильных."""
    aspects = {}  # id -> {"name":…, "statuses": [по прогонам]}
    for rep in reports:
        for v in rep["verdicts"]:
            a = aspects.setdefault(v["id"], {"name": v["name"], "statuses": [],
                                             "applicable": []})
            a["statuses"].append(v["status"])
            a["applicable"].append(bool(v["applicable"]))
    for a in aspects.values():
        a["stable"] = len(set(a["statuses"])) == 1
    n = len(aspects)
    stable = sum(1 for a in aspects.values() if a["stable"])
    return {"n_runs": len(reports), "n_aspects": n, "n_stable": stable,
            "stable_share": round(stable / n, 4) if n else 0.0,
            "aspects": {str(k): aspects[k] for k in sorted(aspects)}}


def main() -> int:
    paths = sys.argv[1:]
    if len(paths) < 2:
        print("нужно минимум два отчёта одного макета")
        return 1
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    shas = {r["meta"].get("source_sha256") for r in reports}
    if len(shas) != 1:
        print(f"отчёты от разных макетов (sha: {shas}) — сравнение бессмысленно")
        return 1

    table = stability_table(reports)
    cfg = load_config()
    out = ROOT / cfg["eval"]["stability_metrics"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, ensure_ascii=False, indent=1),
                   encoding="utf-8")

    print(f"Прогонов: {table['n_runs']}, аспектов: {table['n_aspects']}, "
          f"стабильных: {table['n_stable']} ({table['stable_share']:.0%})")
    for aid, a in table["aspects"].items():
        mark = "✅" if a["stable"] else "🔀"
        print(f"  {mark} {aid:>2} {a['name'][:34]:34s} {' | '.join(a['statuses'])}")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
