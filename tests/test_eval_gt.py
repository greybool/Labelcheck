"""Тесты генерации эталона и прогона метрик (День 7) — без API.

Проверяются: валидатор вопросов (запрет утечки адресов пунктов — иначе BM25
находит пункт по номеру и метрики завышаются), детерминизм и квоты
стратифицированной выборки, расширение accepted_chunk_ids по part-частям,
арифметика Hit@K/MRR, слияние выдач при rewriting, иммутабельность
снапшотов и resume-журнал.

Запуск из корня репозитория:  python tests/test_eval_gt.py
(совместим и с pytest: pytest tests/)
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from evaluation import generate_gt as G
from evaluation import run_retrieval_eval as E
from evaluation.matcher import category_of, load_index, rank_of_record
from labelcheck.retrieval import CHUNKS_PATH, load_config

CFG = load_config()
ECFG = CFG["eval"]

with open(CHUNKS_PATH, encoding="utf-8") as f:
    CHUNKS = [json.loads(line) for line in f]
INDEX = load_index(CHUNKS_PATH)


# ── валидатор вопросов ───────────────────────────────────────────────────────

def test_validator_catches_address_leaks():
    """Вопросы с адресами норм и ссылками на фрагмент бракуются."""
    bad = [
        "Что требует п. 4 о наименовании компонентов?",
        "Какие сведения перечислены в пункте 14 для аллергенов?",
        "Что говорит ТР ТС 022/2011 о маркировке?",
        "Какие добавки перечислены в приложении 2?",
        "Каковы требования статьи 9 к добавкам?",
        "Что сказано в части 4.4 о составе?",
        "Какие требования установлены настоящим регламентом к упаковке?",
        "Что перечислено в данном пункте?",
        "Какие продукты указаны в разделе XI?",
        "Как формируется наименование с учетом ограничений подпунктов «а»–«в»?",
        "Какие требования перечислены в пунктах «а» и «б»?",
    ]
    for q in bad:
        assert G.question_problems(q), f"пропущен негодный вопрос: {q!r}"


def test_validator_catches_hidden_deictics():
    """Скрытые ссылки на фрагмент (разбор промахов v1): «этот/такой/здесь»,
    оформление таблиц, каталожные номера — вопрос не самодостаточен."""
    bad = [
        "Сколько каррагинана можно в такой продукции?",
        "Как называется этот ароматический ингредиент по-химически?",
        "Какие нормы по остаткам лекарств здесь были установлены?",
        "В каких напитках из этой группы делают бодрящие напитки?",
        "Какие виды рыбы в таблице помечены как «н/д» без данных?",
        "Какой номер у 1,4-диметоксибензола в химической базе веществ?",
        "Какое химическое наименование указано для вещества с номером 3752 933?",
        "Что означает пометка для проверки остатков левомицетина?",
    ]
    for q in bad:
        assert G.question_problems(q), f"пропущен несамодостаточный вопрос: {q!r}"


def test_validator_accepts_clean_questions():
    """Нормальные вопросы проходят без замечаний (идиома «при этом» — не ссылка)."""
    good = [
        "Какая температура должна быть у мороженой рыбы при хранении?",
        "Нужно ли указывать функциональный класс добавки Е451 в составе?",
        "Как указывается мясо птицы механической обвалки в составе продукта?",
        "Можно ли писать «без ГМО», и какие условия при этом должны соблюдаться?",
    ]
    for q in good:
        assert not G.question_problems(q), f"забракован годный вопрос: {q!r}"


def test_validator_rejects_short_and_nonquestion():
    """Обрывки и утверждения без вопросительного знака — не вопросы."""
    assert G.question_problems("Что это?")           # короче 15 символов
    assert G.question_problems("Маркировка наносится на упаковку продукции.")
    assert G.question_problems(123)                   # не строка


def test_normalize_question_dedup_key():
    """Ключ дедупа не зависит от регистра, ё и пунктуации."""
    a = G.normalize_question("Какая глазурь у мороженой рыбы?")
    b = G.normalize_question("какая  глазурь у мороженой рыбы")
    assert a == b


# ── выборка ──────────────────────────────────────────────────────────────────

def test_sample_deterministic():
    """Один seed — одна и та же выборка (воспроизводимость эталона)."""
    s1, _ = G.build_sample(CHUNKS, ECFG)
    s2, _ = G.build_sample(CHUNKS, ECFG)
    assert s1 == s2 and len(s1) > 0


def test_sample_quotas_and_floors():
    """Квоты категорий соблюдены, полы по регламентам работают."""
    selected, _ = G.build_sample(CHUNKS, ECFG)
    by_cat, by_cat_reg = {}, {}
    for pos in selected:
        c = CHUNKS[pos]
        assert len(c["text"]) >= ECFG["min_chunk_chars"]
        cat = category_of(c)
        by_cat[cat] = by_cat.get(cat, 0) + 1
        by_cat_reg[(cat, c["regulation_id"])] = \
            by_cat_reg.get((cat, c["regulation_id"]), 0) + 1
    for cat, quota in ECFG["quotas"].items():
        if cat in by_cat:
            assert by_cat[cat] <= quota, f"{cat}: перебор квоты"
    # пол: в body_clause чанки есть у всех 7 регламентов
    avail = {}
    for c in CHUNKS:
        if category_of(c) == "body_clause" and len(c["text"]) >= ECFG["min_chunk_chars"]:
            avail[c["regulation_id"]] = avail.get(c["regulation_id"], 0) + 1
    for reg, have in avail.items():
        expected = min(ECFG["min_per_regulation"], have)
        got = by_cat_reg.get(("body_clause", reg), 0)
        assert got >= expected, f"body_clause/{reg}: {got} < пола {expected}"


def test_allocate_math():
    """Распределение квоты: полы, пропорция, детерминизм, без переборов."""
    alloc = G.allocate(10, {"a": 100, "b": 3, "c": 50}, 4)
    assert sum(alloc.values()) == 10
    assert alloc["b"] == 3                       # весь маленький регламент
    assert all(alloc[r] <= n for r, n in {"a": 100, "b": 3, "c": 50}.items())
    assert alloc == G.allocate(10, {"a": 100, "b": 3, "c": 50}, 4)


# ── accepted_chunk_ids ───────────────────────────────────────────────────────

def _find_appendix_part_chunk():
    for pos, c in enumerate(CHUNKS):
        if category_of(c) == "appendix_clause" and c.get("part") == 2:
            return pos
    raise AssertionError("в корпусе нет appendix-чанка с part=2")


def test_part_run_covers_all_parts():
    """part_run находит все части того же пункта приложения (реальный корпус)."""
    pos = _find_appendix_part_chunk()
    run = G.part_run(pos, CHUNKS)
    assert CHUNKS[pos]["chunk_id"] in run and len(run) >= 2
    clauses = {(CHUNKS[i]["regulation_id"], CHUNKS[i].get("appendix"),
                CHUNKS[i].get("clause"))
               for i, c in enumerate(CHUNKS) if c["chunk_id"] in run}
    assert len(clauses) == 1, "в run попали чанки другого пункта"


def test_part_run_single_for_whole_chunk():
    """Чанк без part-разрезки — run из одного самого себя."""
    pos = next(i for i, c in enumerate(CHUNKS) if not c.get("part"))
    assert G.part_run(pos, CHUNKS) == [CHUNKS[pos]["chunk_id"]]


def test_part_run_separate_runs_not_merged():
    """Два разных «пункта 1» одного приложения (нумерация перезапускается)
    не склеиваются: их part-цепочки не смежны."""
    meta = {"regulation_id": "r", "appendix": "1", "section": None,
            "subsection": None, "clause": "1"}
    fake = [dict(meta, chunk_id="a1", part=1), dict(meta, chunk_id="a2", part=2),
            dict(meta, chunk_id="x", clause="2", part=None),
            dict(meta, chunk_id="b1", part=1), dict(meta, chunk_id="b2", part=2)]
    assert G.part_run(0, fake) == ["a1", "a2"]
    assert G.part_run(3, fake) == ["b1", "b2"]


def test_rank_of_record_uses_accepted_ids():
    """Соседняя part-часть пункта приложения засчитывается только через
    явное поле accepted_chunk_ids (реальный корпус)."""
    pos = _find_appendix_part_chunk()
    run = G.part_run(pos, CHUNKS)
    me, sibling = CHUNKS[pos]["chunk_id"], \
        next(cid for cid in run if cid != CHUNKS[pos]["chunk_id"])
    rec = {"chunk_id": me, "accepted_chunk_ids": run}
    assert rank_of_record(rec, [sibling], INDEX) == 1
    strict = {"chunk_id": me, "accepted_chunk_ids": [me]}
    assert rank_of_record(strict, [sibling], INDEX) is None


# ── генерация: разбор ответа и resume ────────────────────────────────────────

def test_parse_questions_caps_and_types():
    """Лишние вопросы обрезаются по конфигу, не-список — громкая ошибка."""
    raw = json.dumps({"verbatim": ["Вопрос один?", "Два?", "Три лишний?"],
                      "paraphrase": ["Бытовой?"]})
    out = G.parse_questions(raw, 2, 1)
    assert len(out["verbatim"]) == 2 and len(out["paraphrase"]) == 1
    try:
        G.parse_questions(json.dumps({"verbatim": "не список"}), 2, 1)
        raise AssertionError("не-список прошёл")
    except ValueError:
        pass


def test_resume_journal_roundtrip():
    """Журнал: записанные чанки видны при resume, порядок не важен."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "partial.jsonl"
        rows = [{"chunk_id": "a", "records": [], "barren": True},
                {"chunk_id": "b", "records": [{"question": "q"}], "barren": False}]
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                     encoding="utf-8")
        done = G.load_partial(p)
        assert set(done) == {"a", "b"} and done["a"]["barren"]
        assert G.load_partial(Path(td) / "нет.jsonl") == {}


# ── прогон метрик ────────────────────────────────────────────────────────────

def test_metrics_math():
    """Hit@K и MRR на синтетических рангах, руками пересчитано."""
    m = E.metrics_of([1, None, 4, 10])
    assert m["n"] == 4
    assert m["hit@1"] == 0.25 and m["hit@3"] == 0.25
    assert m["hit@5"] == 0.5 and m["hit@10"] == 0.75
    assert abs(m["mrr"] - (1 + 0 + 0.25 + 0.1) / 4) < 1e-9


class _StubRetriever:
    """BM25/vector отдают заготовленные ранги — слияние проверяем без индексов."""

    def __init__(self, per_query):
        self.per_query = per_query

    def bm25_search(self, q, k, regulation=None):
        return [(cid, 1.0) for cid in self.per_query[q]["bm25"][:k]]

    def vector_search(self, q, k, regulation=None):
        return [(cid, 1.0) for cid in self.per_query[q]["vector"][:k]]


def test_run_search_single_vs_fused():
    """Один запрос — нативная выдача метода; несколько — RRF-слияние."""
    scfg = {"candidates": 20, "top_k": 10, "rrf_k": 60}
    stub = _StubRetriever({
        "q": {"bm25": ["a", "b"], "vector": ["c", "a"]},
        "rw": {"bm25": ["b", "d"], "vector": ["d", "e"]},
    })
    assert E.run_search(stub, "bm25", ["q"], scfg) == ["a", "b"]
    fused = E.run_search(stub, "bm25", ["q", "rw"], scfg)
    assert fused[0] in ("a", "b") and "d" in fused  # выдачи слились
    hybrid = E.run_search(stub, "hybrid", ["q"], scfg)
    assert hybrid[0] == "a"  # 'a' есть в обеих выдачах — RRF поднимает его


def test_write_run_immutable():
    """Повторная запись того же run_id — громкая ошибка, не перезапись."""
    with tempfile.TemporaryDirectory() as td:
        runs = Path(td)
        E.write_run(runs, "r1", {"m": 1}, [{"qi": 0}])
        assert (runs / "r1" / "results.jsonl").exists()
        try:
            E.write_run(runs, "r1", {"m": 2}, [])
            raise AssertionError("снапшот перезаписался")
        except FileExistsError:
            pass


def test_eval_rewrite_cfg_isolated():
    """Кэш переформулировок эталона отделён от аспектного, исходный конфиг цел."""
    rcfg = E.eval_rewrite_cfg(CFG)
    assert rcfg["rewrite"]["cache"] == ECFG["rewrites_cache"]
    assert rcfg["rewrite"]["cache"] != CFG["rewrite"]["cache"]
    assert rcfg["rewrite"]["enabled"] is True
    assert CFG["rewrite"]["cache"] != ECFG["rewrites_cache"] or False
    assert CFG["eval"]["rewrites_cache"] == ECFG["rewrites_cache"]


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
