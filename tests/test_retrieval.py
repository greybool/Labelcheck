"""Тесты поисковой части без API: токенизация, RRF, BM25 на реальном корпусе.

Векторный и гибридный поиск требуют ключа OpenAI и кэша векторов — они
проверяются смоук-скриптом (evaluation/smoke_retrieval.py) и метриками
Дня 7; здесь только то, что бесплатно и детерминировано.

Запуск из корня репозитория:  python tests/test_retrieval.py
(совместим и с pytest: pytest tests/)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from labelcheck.retrieval import Retriever, Tokenizer, load_config, rrf_fuse

CFG = load_config()
TOKENIZER = Tokenizer(CFG["bm25"])
RETRIEVER = Retriever(CFG)  # без openai_client: BM25 доступен, векторы нет


# --- Токенизация ---

def test_stemming_unifies_forms():
    """Падежные формы сводятся к одной основе: «глазури» находит «глазурь»."""
    assert TOKENIZER("глазури") == TOKENIZER("глазурью") == TOKENIZER("глазурь")


def test_latin_e_code_normalized():
    """E511 латиницей и Е511 кириллицей — один токен (в корпусе оба алфавита)."""
    assert TOKENIZER("E511") == TOKENIZER("Е511")
    assert TOKENIZER("e243") == TOKENIZER("Е243")


def test_e_code_not_mangled():
    """Стеммер не калечит Е-коды: цифры остаются на месте."""
    token = TOKENIZER("Е243")[0]
    assert token.endswith("243"), token


def test_plain_e_not_normalized():
    """Одинокая буква «e» без цифр не превращается в Е-код."""
    assert TOKENIZER("e") != TOKENIZER("е243")


# --- RRF ---

def test_rrf_agreement_wins():
    """Документ, попавший в обе выдачи, обгоняет документы из одной."""
    fused = rrf_fuse([["a", "b", "c"], ["a", "c", "b"]], rrf_k=60, top_k=3)
    assert fused[0][0] == "a"


def test_rrf_uses_ranks_not_scores():
    """Балл зависит только от позиций: 1/(k+место), суммируется по выдачам."""
    fused = dict(rrf_fuse([["x"], ["x"]], rrf_k=60, top_k=1))
    assert abs(fused["x"] - 2 / 61) < 1e-9


def test_rrf_top_k_limits():
    fused = rrf_fuse([["a", "b", "c", "d"]], rrf_k=60, top_k=2)
    assert len(fused) == 2


# --- BM25 на реальном корпусе ---

def test_bm25_finds_glaze_clause():
    """«масса глазури рыбной продукции» → п.33 ТР ЕАЭС 040 в топ-5."""
    top = [cid for cid, _ in RETRIEVER.bm25_search("масса глазури рыбной продукции", k=5)]
    by_id = {c["chunk_id"]: c for c in RETRIEVER.chunks}
    assert any(by_id[cid]["regulation_id"] == "ТР ЕАЭС 040/2016"
               and by_id[cid].get("clause") == "33" for cid in top), top


def test_bm25_finds_e_code_by_latin_query():
    """Запрос с латинским E-кодом находит кириллический Е-код в 029."""
    top = [cid for cid, _ in RETRIEVER.bm25_search("консервант E243", k=5)]
    assert any(cid.startswith("tr_ts_029_2012") for cid in top), top


def test_bm25_regulation_filter():
    """Фильтр по регламенту не пропускает чужие документы."""
    top = RETRIEVER.bm25_search("маркировка продукции",
                                k=10, regulation="ТР ТС 022/2011")
    assert top, "фильтр вернул пустую выдачу"
    assert all(RETRIEVER.regulation_of[cid] == "ТР ТС 022/2011" for cid, _ in top)


def test_bm25_scores_descending():
    top = RETRIEVER.bm25_search("срок годности пищевой продукции", k=10)
    scores = [s for _, s in top]
    assert scores == sorted(scores, reverse=True)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"✅ {name}: {fn.__doc__ or ''}".strip())
        except Exception as e:  # и AssertionError, и сбой самого поиска
            failed += 1
            print(f"❌ {name}: {e or fn.__doc__}")
    print(f"\n{len(tests) - failed}/{len(tests)} проверок пройдено")
    sys.exit(1 if failed else 0)
