"""CLI полной проверки макета: PDF или готовый layout-JSON → отчёт.

  python -m labelcheck.check <макет.pdf>          # vision + вердикты
  python -m labelcheck.check <layout.json>        # вердикты по готовому JSON

Выход: data/reports/<имя>.json + <имя>.md (папка в .gitignore — тексты
реальных макетов). Сводка и расход токенов печатаются в консоль.
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from labelcheck.retrieval import ROOT, Retriever, load_config
from labelcheck.verdict import (STATUS_COMPLIANT, STATUS_MANUAL,
                                STATUS_VIOLATION, check_layout, render_markdown)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Проверка макета по 21 аспекту")
    ap.add_argument("path", help="PDF-макет или layout-JSON (выход vision)")
    ap.add_argument("-o", "--out-dir", default=None,
                    help="куда класть отчёты (по умолчанию config → verdict.reports_dir)")
    ap.add_argument("--categories", default="auto",
                    help="категорийные регламенты («кнопки»): auto — детект по "
                         "маркерам; none — только горизонтальные; либо список "
                         "через запятую из poultry,meat,fish "
                         "(например --categories poultry)")
    args = ap.parse_args(argv)

    if args.categories == "auto":
        override = None
    elif args.categories == "none":
        override = set()
    else:
        override = {c.strip() for c in args.categories.split(",") if c.strip()}
        unknown = override - {"poultry", "meat", "fish"}
        if unknown:
            ap.error(f"неизвестные категории: {', '.join(sorted(unknown))}")

    load_dotenv()
    cfg = load_config()
    client = OpenAI()
    path = Path(args.path)

    if path.suffix.lower() == ".json":
        layout = json.loads(path.read_text(encoding="utf-8"))
    else:
        from labelcheck import vision  # импорт здесь: тянет pypdfium2/PIL
        layout_path = Path("data/layouts") / (path.stem + ".json")
        layout = vision.analyze(path, out_path=layout_path)
        print(f"layout: {layout_path}")

    print("Строю индексы и коллекцию векторов (в памяти, может занять минуты)…")
    retriever = Retriever(cfg, openai_client=client)
    report = check_layout(layout, retriever, client, cfg,
                          categories_override=override)

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / cfg["verdict"]["reports_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(layout["meta"].get("source_pdf") or path.stem).stem
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    icons = {STATUS_VIOLATION: "🔴", STATUS_MANUAL: "🟡", STATUS_COMPLIANT: "🟢"}
    for v in report["verdicts"]:
        icon = "⚪" if not v["applicable"] else icons[v["status"]]
        print(f"{icon} {v['id']:>2}. {v['name']}: "
              f"{'не применимо' if not v['applicable'] else v['status']}"
              f"{' [понижено]' if v['downgraded_reason'] else ''}")
    for block in report["other_remarks"]:
        print(f"·  {block['id']}. {block['name']}: {len(block['items'])} замечаний")
    for model, d in report["meta"]["tokens"].items():
        print(f"токены {model}: {d['prompt']}+{d['completion']} ({d['calls']} вызовов)")
    print(f"отчёт: {md_path} (+ .json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
