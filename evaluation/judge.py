"""LLM-as-judge (День 8, ТЗ §5.2, подход 1): аудит вердиктов CHEAP-моделью.

Запуск из корня репозитория:
  python evaluation/judge.py data/reports/<отчёт1>.json [<отчёт2>.json …]

Судья (CHEAP — модель, отличная от MAIN-генератора вердиктов) получает для
каждого регламентного вердикта: аспект, факты макета, вердикт с цитатами и
полные тексты процитированных пунктов — и отвечает по рубрике ТЗ:
обоснованность (reasoning_supported), релевантность цитат
(citations_relevant), выдуманные пункты (invented_clauses).

Приватность (правило 7, публичный репозиторий): сырые ответы судьи содержат
тексты реального макета — они пишутся в data/reports/judge/ (в .gitignore).
В git уходит только evaluation/metrics/llm_eval.json — счётчики по аспектам
без единой строки текста макета.

Аспект 17 (шрифт) не судится: его вердикт детерминированный, без LLM.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from labelcheck.retrieval import CHUNKS_PATH, load_config

MAX_CHUNK_CHARS = 3500   # текст процитированного пункта в промпте судьи:
                         # полный чанк (max_chars нарезки = 3500). День 9:
                         # лимит 1500 обрезал процитированную норму (цитата
                         # аспекта 20 начиналась с позиции 1313) — судья
                         # записывал «выдуманный пункт» на ровном месте
MAX_FACTS_CHARS = 12000  # факты макета в промпте судьи (хвост режется)


def load_chunk_texts() -> dict[str, str]:
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return {c["chunk_id"]: c["text"]
                for c in (json.loads(line) for line in f)}


def collect_layout_facts(layout: dict) -> str:
    """Дословные тексты регионов макета (как их видел вердикт-пайплайн)."""
    parts = []
    for r in layout.get("regions", []):
        if r.get("kind") == "technical":
            continue
        txt = (r.get("text") or "").strip()
        if txt:
            parts.append(f"[{r.get('kind')}/{r.get('lang')}] {txt}")
    return "\n".join(parts)[:MAX_FACTS_CHARS]


def basis_context_texts(aspect: dict, categories: set[str], chunks: list[dict],
                        cfg: dict, facts: str,
                        aspects_data: dict) -> list[str]:
    """Детерминированная часть контекста, которую видела вердикт-модель:
    basis-пункты + пункты области применения + Е-код-lookup. Ретрив-хвост
    (retrieval_extra) меняется от прогона к прогону и здесь не воссоздаётся —
    процитированные из него пункты судья и так получает отдельным блоком."""
    from labelcheck.verdict import gather_context
    scope_basis = [b for cat in sorted(categories)
                   for b in aspects_data.get("category_scope_basis", {}).get(cat, [])]
    basis, extra = gather_context(aspect, categories, chunks,
                                  search_fn=lambda q, regs: [], cfg=cfg,
                                  facts_text=facts, scope_basis=scope_basis)
    return [f"--- {c['chunk_id']} ---\n{c['text']}" for c in basis + extra]


def build_judge_user(verdict: dict, facts: str, chunk_texts: dict[str, str],
                     basis_texts: list[str] | None = None,
                     check_text: str = "") -> str:
    """Пользовательский промпт судьи. check_text — доменные правила проверки
    аспекта из aspects.yaml (день 9, решение Сергея): без них судья наказывал
    вердикты за следование нашим же правилам (даты-шаблоны DD/MM/YYYY)."""
    cited = []
    for c in verdict.get("citations", []):
        text = chunk_texts.get(c["chunk_id"], "")[:MAX_CHUNK_CHARS]
        cited.append(f"--- {c['chunk_id']} ---\n{text}")
    payload = {
        "аспект": verdict["name"],
        "статус": verdict["status"],
        "применимость": verdict["applicable"],
        "объяснение": verdict.get("explanation", ""),
        "цитаты": [{"chunk_id": c["chunk_id"], "quote": c["quote"]}
                   for c in verdict.get("citations", [])],
    }
    user = f"ФАКТЫ МАКЕТА:\n{facts}\n\n"
    if check_text.strip():
        user += ("ПРАВИЛА ПРОВЕРКИ АСПЕКТА (заданы владельцем системы, "
                 "часть постановки задачи):\n" + check_text.strip() + "\n\n")
    user += (f"ВЕРДИКТ СИСТЕМЫ:\n{json.dumps(payload, ensure_ascii=False, indent=1)}\n\n"
             f"ТЕКСТЫ ПРОЦИТИРОВАННЫХ ПУНКТОВ:\n" +
             ("\n".join(cited) if cited else "(цитат нет)"))
    if basis_texts is not None:
        user += ("\n\nПОЛНЫЙ НАБОР НОРМ, КОТОРЫЙ ВИДЕЛА СИСТЕМА:\n"
                 + "\n".join(basis_texts))
    return user


def judge_verdict(client, model: str, prompt: str, verdict: dict,
                  facts: str, chunk_texts: dict[str, str],
                  basis_texts: list[str] | None = None,
                  check_text: str = "") -> dict:
    """Один вердикт → ответ судьи (3 попытки на сетевые сбои и битый JSON)."""
    user = build_judge_user(verdict, facts, chunk_texts, basis_texts, check_text)
    last = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": prompt},
                          {"role": "user", "content": user}])
            data = json.loads(resp.choices[0].message.content)
            out = {"reasoning_supported": bool(data["reasoning_supported"]),
                   "citations_relevant": (None if data.get("citations_relevant")
                                          is None else bool(data["citations_relevant"])),
                   "invented_clauses": bool(data["invented_clauses"]),
                   "comment": str(data.get("comment", ""))[:600],
                   "tokens": resp.usage.total_tokens}
            return out
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"судья не ответил за 3 попытки: {last}")


def aggregate(raw_by_run: dict[str, dict]) -> dict:
    """Сырые ответы судьи по прогонам → счётчики для git (без текстов)."""
    per_aspect = defaultdict(lambda: {"n": 0, "supported": 0, "relevant_ok": 0,
                                      "relevant_judged": 0, "invented": 0})
    total = {"n": 0, "supported": 0, "invented": 0,
             "relevant_ok": 0, "relevant_judged": 0}
    for run_label, verdicts in raw_by_run.items():
        for aid, j in verdicts.items():
            a = per_aspect[aid]
            a["n"] += 1
            total["n"] += 1
            if j["reasoning_supported"]:
                a["supported"] += 1
                total["supported"] += 1
            if j["invented_clauses"]:
                a["invented"] += 1
                total["invented"] += 1
            if j["citations_relevant"] is not None:
                a["relevant_judged"] += 1
                total["relevant_judged"] += 1
                if j["citations_relevant"]:
                    a["relevant_ok"] += 1
                    total["relevant_ok"] += 1
    return {"per_aspect": dict(sorted(per_aspect.items(),
                                      key=lambda kv: int(kv[0]))),
            "total": total, "runs": sorted(raw_by_run)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+",
                        help="JSON-отчёты вердикт-пайплайна")
    parser.add_argument("--layout", required=True,
                        help="layout-JSON макета (источник фактов)")
    parser.add_argument("--with-basis", action="store_true",
                        help="судья видит basis-контекст, который видела "
                             "вердикт-модель (режим with_basis)")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    import os

    from openai import OpenAI

    from labelcheck.aspects import load_aspects
    cfg = load_config()
    ecfg = cfg["eval"]
    mode = "with_basis" if args.with_basis else "strict"
    model = os.environ[ecfg["judge_model_env"]]
    prompt = ecfg["judge_basis_prompt" if args.with_basis else "judge_prompt"]
    out_dir = ROOT / ecfg["judge_out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    layout = json.loads(Path(args.layout).read_text(encoding="utf-8"))
    facts = collect_layout_facts(layout)
    chunk_texts = load_chunk_texts()
    aspects_data = load_aspects()
    by_id = {a["id"]: a for a in aspects_data["aspects"]}
    client = OpenAI()

    suffix = ".judge_basis.json" if args.with_basis else ".judge.json"
    raw_by_run, tokens = {}, 0
    for i, rpath in enumerate(args.reports, start=1):
        report = json.loads(Path(rpath).read_text(encoding="utf-8"))
        categories = set(report["meta"].get("categories", {}))
        label = f"run{i}"
        out_file = out_dir / (Path(rpath).stem + suffix)
        if out_file.exists():  # уже судили — API не зовём (resume)
            raw_by_run[label] = json.loads(out_file.read_text(encoding="utf-8"))
            print(f"{rpath}: ответы судьи уже есть — {out_file.name}")
            continue
        judged = {}
        for v in report["verdicts"]:
            if v["key"] == "font_size":
                continue  # детерминированный вердикт без LLM — судить нечего
            basis_texts = None
            if args.with_basis:
                basis_texts = basis_context_texts(
                    by_id[v["id"]], categories,
                    _CHUNKS_CACHE.setdefault("chunks", _load_chunks()),
                    cfg, facts, aspects_data)
            judged[str(v["id"])] = judge_verdict(
                client, model, prompt, v, facts, chunk_texts, basis_texts,
                check_text=by_id[v["id"]].get("check", ""))
            tokens += judged[str(v["id"])]["tokens"]
            print(f"\r{rpath}: аспектов оценено {len(judged)}", end="", flush=True)
        print()
        out_file.write_text(json.dumps(judged, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        raw_by_run[label] = judged

    summary = aggregate(raw_by_run)
    summary["judge_model"] = model

    out = ROOT / ecfg["llm_eval_metrics"]
    out.parent.mkdir(parents=True, exist_ok=True)
    combined = (json.loads(out.read_text(encoding="utf-8"))
                if out.exists() else {})
    if "per_aspect" in combined:  # файл старого формата (только strict)
        combined = {"strict": combined}
    combined[mode] = summary
    out.write_text(json.dumps(combined, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    t = summary["total"]
    print(f"Судья ({model}, режим {mode}), прогонов: {len(raw_by_run)}, "
          f"токены: {tokens:,}")
    print(f"  обоснованность: {t['supported']}/{t['n']}")
    print(f"  цитаты релевантны: {t['relevant_ok']}/{t['relevant_judged']}")
    print(f"  выдуманные пункты: {t['invented']}/{t['n']}")
    print(f"Счётчики → {out} (сырые ответы судьи — в {out_dir}/, не в git)")
    return 0


_CHUNKS_CACHE: dict = {}


def _load_chunks() -> list[dict]:
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    sys.exit(main())
