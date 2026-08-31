"""Тесты Дня 8 без API: судья (агрегация), стабильность, cosine-метрика,
шаблоны и метрика vision-оценки.

Запуск из корня репозитория:  python tests/test_eval_llm.py
(совместим и с pytest: pytest tests/)
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from evaluation import answer_similarity as S
from evaluation import judge as J
from evaluation import stability as ST
from evaluation import vision_gt as VG

# ── судья ────────────────────────────────────────────────────────────────────

def test_judge_aggregate_counts():
    """Счётчики судьи: обоснованность, релевантность (с null), выдумки."""
    raw = {"run1": {"1": {"reasoning_supported": True, "citations_relevant": True,
                          "invented_clauses": False, "comment": ""},
                    "2": {"reasoning_supported": False, "citations_relevant": None,
                          "invented_clauses": True, "comment": ""}},
           "run2": {"1": {"reasoning_supported": True, "citations_relevant": False,
                          "invented_clauses": False, "comment": ""}}}
    out = J.aggregate(raw)
    t = out["total"]
    assert t["n"] == 3 and t["supported"] == 2 and t["invented"] == 1
    assert t["relevant_judged"] == 2 and t["relevant_ok"] == 1  # null не судится
    assert out["per_aspect"]["1"]["supported"] == 2
    assert list(out["per_aspect"]) == ["1", "2"]  # сортировка по номеру


def test_judge_facts_skip_technical_and_cap():
    """Факты для судьи: technical-регионы выброшены, длина ограничена."""
    layout = {"regions": [
        {"kind": "technical", "lang": "ru", "text": "PANTONE 485"},
        {"kind": "composition", "lang": "ru", "text": "Состав: мука, вода"},
        {"kind": "other_text", "lang": "ko", "text": "х" * 20000},
    ]}
    facts = J.collect_layout_facts(layout)
    assert "PANTONE" not in facts and "Состав: мука" in facts
    assert len(facts) <= J.MAX_FACTS_CHARS

# ── стабильность ─────────────────────────────────────────────────────────────

def _rep(statuses):
    return {"verdicts": [{"id": i + 1, "name": f"аспект {i + 1}", "status": s,
                          "applicable": True}
                         for i, s in enumerate(statuses)]}


def test_stability_math():
    """Доля стабильных аспектов и список статусов по прогонам."""
    reports = [_rep(["соответствует", "возможное нарушение"]),
               _rep(["соответствует", "требует ручной проверки"])]
    t = ST.stability_table(reports)
    assert t["n_runs"] == 2 and t["n_aspects"] == 2 and t["n_stable"] == 1
    assert t["stable_share"] == 0.5
    assert t["aspects"]["1"]["stable"] and not t["aspects"]["2"]["stable"]
    assert t["aspects"]["2"]["statuses"] == ["возможное нарушение",
                                             "требует ручной проверки"]

# ── cosine-метрика ───────────────────────────────────────────────────────────

def test_similarity_split_hit_miss():
    """Косинус топ-1↔цель делится на попадания и промахи."""
    e = lambda *xs: np.asarray(xs, dtype=np.float32) / np.linalg.norm(xs)
    vectors = {"gold": e(1, 0), "same": e(1, 0), "near": e(1, 1), "far": e(0, 1)}
    index = {cid: {"key": ("chunk", cid)} for cid in vectors}
    records = [{"chunk_id": "gold", "accepted_chunk_ids": ["gold"]},
               {"chunk_id": "gold", "accepted_chunk_ids": ["gold"]}]
    results = [{"retrieved": ["gold", "far"]},   # попадание, cos=1
               {"retrieved": ["near", "far"]}]   # промах, cos≈0.707
    out = S.run_similarity(records, results, index, vectors)
    assert out["when_hit@10"]["n"] == 1 and out["when_hit@10"]["mean"] == 1.0
    assert out["when_miss@10"]["n"] == 1
    assert abs(out["when_miss@10"]["mean"] - 0.7071) < 1e-3

# ── vision-оценка ────────────────────────────────────────────────────────────

def test_vision_template_fields():
    """Шаблон: technical выброшен, verdict предзаполнен «точно»."""
    layout = {"meta": {"source_pdf": "x.pdf", "source_sha256": "abc"},
              "regions": [
                  {"id": "r1", "kind": "composition", "lang": "ru",
                   "text": "Состав: мука", "status": "прочитано"},
                  {"id": "r2", "kind": "technical", "lang": None, "text": "метки"}]}
    t = VG.make_template(layout)
    assert len(t["fields"]) == 1
    f = t["fields"][0]
    assert f["id"] == "r1" and f["verdict"] == "точно" and f["corrected_text"] == ""
    assert t["missed_blocks"] == [] and t["annotated_by"] == ""


def test_vision_score_shares():
    """Метрика ТЗ §5.3: доли точно/искажение/пропущено, пропуски блоков
    считаются пропусками."""
    gt = {"source_sha256": "abc", "missed_blocks": ["состав KO"],
          "fields": [
              {"id": "r1", "lang": "ru", "verdict": "точно"},
              {"id": "r2", "lang": "ru", "verdict": "искажение"},
              {"id": "r3", "lang": "ko", "verdict": "точно"}]}
    out = VG.score([gt])
    assert out["n_fields"] == 3 and out["missed_blocks"] == 1
    assert out["shares"]["точно"] == 0.5           # 2 из 4 (3 поля + 1 блок)
    assert out["shares"]["искажение"] == 0.25
    assert out["shares"]["пропущено"] == 0.25
    assert out["by_language"]["ru"]["искажение"] == 1


def test_vision_score_rejects_unknown_verdict():
    """Опечатка в verdict — громкая ошибка, а не тихий пропуск поля."""
    gt = {"source_sha256": "abc", "missed_blocks": [],
          "fields": [{"id": "r1", "lang": "ru", "verdict": "тошно"}]}
    try:
        VG.score([gt])
        raise AssertionError("неизвестный verdict прошёл")
    except ValueError:
        pass


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
