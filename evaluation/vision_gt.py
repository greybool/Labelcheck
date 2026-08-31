"""Оценка vision-извлечения (День 8, ТЗ §5.3): шаблон эталона + метрика.

Два режима, запуск из корня репозитория:

  python evaluation/vision_gt.py template data/layouts/<макет>.json
      → data/vision_gt/<макет>.gt.json — заготовка эталонной разметки:
        каждый регион с текстом vision и полем verdict, предзаполненным
        "точно". Сергей правит руками, глядя на исходный макет:
        verdict: "точно" | "искажение" | "пропуск-части" (регион найден, но
        часть текста не прочитана) — и вписывает corrected_text там, где
        vision ошибся; блоки, целиком пропущенные обзором, добавляет в
        missed_blocks (["наименование KO", …]). Повторный запуск шаблон
        НЕ перезаписывает (разметка — ручной труд).

  python evaluation/vision_gt.py score
      → по всем размеченным data/vision_gt/*.gt.json считает метрику ТЗ:
        доля полей, извлечённых точно / с искажением / пропущенных
        (пропуски = "пропуск-части" + missed_blocks), с разбивкой по языкам.
        Числа → evaluation/metrics/vision_eval.json (в git — только
        счётчики; сами разметки с текстами макетов остаются в
        data/vision_gt/, папка в .gitignore).

Прогон повторяется после любого изменения промпта vision или схемы нарезки
(ТЗ §5.3): эталон привязан к макету через SHA256 исходного PDF.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from labelcheck.retrieval import load_config

VERDICTS = ("точно", "искажение", "пропуск-части")


def make_template(layout: dict) -> dict:
    """layout-JSON → заготовка эталона для ручной правки."""
    fields = []
    for r in layout.get("regions", []):
        if r.get("kind") == "technical":
            continue
        fields.append({
            "id": r["id"],
            "kind": r.get("kind"),
            "lang": r.get("lang"),
            "vision_text": r.get("text") or "",
            "vision_status": r.get("status"),
            "verdict": "точно",       # ← правится руками
            "corrected_text": "",     # ← дословный текст с макета при искажении
            "comment": "",
        })
    return {
        "source_pdf": layout.get("meta", {}).get("source_pdf"),
        "source_sha256": layout.get("meta", {}).get("source_sha256"),
        "instruction": ("Для каждого поля: verdict = точно | искажение | "
                        "пропуск-части; при искажении впиши corrected_text. "
                        "Блоки, которых нет в списке вовсе, добавь в "
                        "missed_blocks строками вида 'состав KO'."),
        "fields": fields,
        "missed_blocks": [],
        "annotated_by": "",           # ← имя разметчика = разметка готова
    }


def score(gt_files: list[dict]) -> dict:
    """Размеченные эталоны → метрика ТЗ §5.3 (только счётчики, без текстов)."""
    layouts = []
    total = defaultdict(int)
    by_lang = defaultdict(lambda: defaultdict(int))
    for gt in gt_files:
        counts = defaultdict(int)
        for f in gt["fields"]:
            v = f.get("verdict", "").strip()
            if v not in VERDICTS:
                raise ValueError(f"{gt.get('source_sha256', '?')[:8]}: "
                                 f"поле {f['id']}: неизвестный verdict {v!r}")
            counts[v] += 1
            total[v] += 1
            by_lang[f.get("lang") or "?"][v] += 1
        counts["пропущено-блоков"] = len(gt.get("missed_blocks", []))
        total["пропущено-блоков"] += counts["пропущено-блоков"]
        layouts.append({"source_sha256": (gt.get("source_sha256") or "")[:12],
                        "fields": sum(counts[v] for v in VERDICTS),
                        **{k: counts[k] for k in
                           (*VERDICTS, "пропущено-блоков")}})
    n_fields = sum(total[v] for v in VERDICTS)
    n_all = n_fields + total["пропущено-блоков"]
    shares = ({"точно": round(total["точно"] / n_all, 4),
               "искажение": round(total["искажение"] / n_all, 4),
               "пропущено": round((total["пропуск-части"]
                                   + total["пропущено-блоков"]) / n_all, 4)}
              if n_all else {})
    return {"n_layouts": len(gt_files), "n_fields": n_fields,
            "missed_blocks": total["пропущено-блоков"], "shares": shares,
            "by_language": {lang: dict(c) for lang, c in sorted(by_lang.items())},
            "per_layout": layouts}


def main() -> int:
    cfg = load_config()
    gt_dir = ROOT / cfg["eval"]["vision_gt_dir"]
    gt_dir.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) >= 3 and sys.argv[1] == "template":
        src = Path(sys.argv[2])
        out = gt_dir / (src.stem + ".gt.json")
        if out.exists():
            print(f"{out} уже существует — разметка руками не перезаписывается")
            return 1
        layout = json.loads(src.read_text(encoding="utf-8"))
        out.write_text(json.dumps(make_template(layout), ensure_ascii=False,
                                  indent=1), encoding="utf-8")
        print(f"Заготовка эталона: {out} "
              f"({sum(1 for _ in json.loads(out.read_text(encoding='utf-8'))['fields'])} полей)")
        return 0

    if len(sys.argv) >= 2 and sys.argv[1] == "score":
        files = sorted(gt_dir.glob("*.gt.json"))
        annotated = []
        for p in files:
            gt = json.loads(p.read_text(encoding="utf-8"))
            if gt.get("annotated_by"):
                annotated.append(gt)
            else:
                print(f"{p.name}: annotated_by пуст — разметка не начата, пропуск")
        if not annotated:
            print("нет размеченных эталонов")
            return 1
        result = score(annotated)
        out = ROOT / "evaluation/metrics/vision_eval.json"
        out.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        print(json.dumps(result["shares"], ensure_ascii=False))
        print(f"→ {out}")
        return 0

    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
