"""Тесты аспектного чек-листа (aspects.yaml) без API.

Главная проверка — test_every_basis_clause_resolves: каждый адрес пункта
из basis обязан находиться в корпусе (data/chunks.jsonl). Выдуманный или
опечатанный пункт роняет тест — якорь против галлюцинаций в самом чек-листе.

Запуск из корня репозитория:  python tests/test_aspects.py
(совместим и с pytest: pytest tests/)
"""

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))



from labelcheck.aspects import (CategoryDetector, basis_matches_chunk,
                                find_basis_chunks, load_aspects,
                                resolve_regulations)
from labelcheck.retrieval import Retriever, load_config
from labelcheck import rewrite

CFG = load_config()
DATA = load_aspects()
ASPECTS = DATA["aspects"]
BY_ID = {a["id"]: a for a in ASPECTS}
RETRIEVER = Retriever(CFG)  # без openai_client: BM25 и корпус доступны
CHUNKS = RETRIEVER.chunks
CORPUS_REGS = {c["regulation_id"] for c in CHUNKS}
DETECT = CategoryDetector(DATA["category_markers"], RETRIEVER.tokenizer)


# --- Структура файла ---

def test_aspect_count_and_ids():
    """21 аспект (19 из ТЗ §4.4 + 2 решением Дня 5), id — 1..21 без дырок."""
    assert len(ASPECTS) == 21, len(ASPECTS)
    assert sorted(a["id"] for a in ASPECTS) == list(range(1, 22))


def test_keys_unique():
    """Машинные имена уникальны (пойдут в логи, метрики, SQLite)."""
    keys = [a["key"] for a in ASPECTS]
    assert len(set(keys)) == len(keys)
    assert all(k.replace("_", "").isascii() and k == k.lower() for k in keys), keys


def test_groups_split():
    """18 и 19 — «прочие замечания», остальные — регламентные (ТЗ §4.4)."""
    for a in ASPECTS:
        expected = "other" if a["id"] in (18, 19) else "regulatory"
        assert a["group"] == expected, (a["id"], a["group"])


def test_other_aspects_do_not_search():
    """Аспекты-«прочие» в регламенты не ходят: ни basis, ни queries."""
    for a in ASPECTS:
        if a["group"] == "other":
            for field in ("regulations", "category_regulations", "basis", "queries"):
                assert field not in a, (a["id"], field)
            assert resolve_regulations(a, {"meat", "fish"}) == []


def test_regulatory_aspects_complete():
    """У каждого регламентного аспекта есть регламенты, основания и запросы."""
    for a in ASPECTS:
        if a["group"] == "regulatory":
            assert a.get("regulations"), a["id"]
            assert a.get("basis"), a["id"]
            assert a.get("queries"), a["id"]
            assert a.get("check"), a["id"]


def test_regulation_names_exist_in_corpus():
    """Все имена регламентов — ровно как regulation_id в chunks.jsonl."""
    for a in ASPECTS:
        named = list(a.get("regulations", []))
        for regs in a.get("category_regulations", {}).values():
            named += regs
        for reg in named:
            assert reg in CORPUS_REGS, (a["id"], reg)


def test_vision_kinds_valid():
    """vision_kinds аспектов — подмножество категорий vision-конфига."""
    kinds = set(CFG["vision"]["kinds"])
    for a in ASPECTS:
        assert set(a.get("vision_kinds", [])) <= kinds, a["id"]


# --- Валидность оснований по корпусу ---

def test_every_basis_clause_resolves():
    """Каждый адрес (включая КАЖДЫЙ пункт из clauses) находится в корпусе."""
    missing = []
    for a in ASPECTS:
        for basis in a.get("basis", []):
            variants = ([{**basis, "clauses": [c]} for c in basis["clauses"]]
                        if "clauses" in basis else [basis])
            for v in variants:
                if not find_basis_chunks(v, CHUNKS):
                    missing.append((a["id"], v))
    assert not missing, missing


def test_basis_regs_covered_by_search_regs():
    """Регламент каждого основания входит в поисковый список аспекта
    (иначе вердикту не из чего цитировать это основание)."""
    for a in ASPECTS:
        searchable = set(resolve_regulations(a, {"meat", "fish"}))
        for basis in a.get("basis", []):
            assert basis["reg"] in searchable, (a["id"], basis["reg"])


def test_article_number_is_delimited():
    """article «6» не матчит «СТАТЬЯ 6_1.» (номер закрыт точкой)."""
    chunk = {"regulation_id": "ТР ТС 021/2011",
             "section": "СТАТЬЯ 6_1. ОТДЕЛЬНЫЕ ТРЕБОВАНИЯ", "clause": None}
    assert not basis_matches_chunk({"reg": "ТР ТС 021/2011", "article": "6"}, chunk)


def test_roman_section_is_delimited():
    """section «X» не матчит разделы «XI.», «XIV.» (римская цифра с точкой)."""
    chunk = {"regulation_id": "ТР ТС 034/2013",
             "section": "XI. ТРЕБОВАНИЯ К МАРКИРОВКЕ", "clause": "107"}
    assert not basis_matches_chunk({"reg": "ТР ТС 034/2013", "section": "X"}, chunk)
    assert basis_matches_chunk({"reg": "ТР ТС 034/2013", "section": "XI"}, chunk)


def test_subsection_number_is_delimited():
    """subsection «4.1» не матчит подразделы «4.11.», «4.12.»."""
    chunk = {"regulation_id": "ТР ТС 022/2011",
             "subsection": "4.11. Требования к указанию сведений о ГМО",
             "clause": "1"}
    assert not basis_matches_chunk(
        {"reg": "ТР ТС 022/2011", "subsection": "4.1"}, chunk)


# --- Детект категорий ---

def test_detect_meat_by_word_form():
    """Падежная форма находит маркер: «свинины» → meat (стемминг)."""
    hits = DETECT("Состав: филе свинины, мука пшеничная, вода, соль")
    assert set(hits) == {"meat"}, hits


def test_detect_fish():
    hits = DETECT("Пельмени с треской и креветками")
    assert set(hits) == {"fish"}, hits


def test_detect_adjective_marker():
    """Однокоренное прилагательное — отдельный маркер: «крабовые палочки»."""
    hits = DETECT("Крабовые палочки замороженные")
    assert set(hits) == {"fish"}, hits


def test_detect_both_categories():
    hits = DETECT("начинка: свинина, сурими, лук")
    assert set(hits) == {"meat", "fish"}, hits


def test_detect_nothing_for_fruit():
    """Манго и овощи не поднимают категорийные регламенты."""
    assert DETECT("Манго замороженное кусочками, сахар") == {}


def test_resolve_regulations_layers():
    """База всегда; категорийные — по детекту; без дублей, база первой."""
    name_aspect = BY_ID[1]
    assert resolve_regulations(name_aspect, set()) == ["ТР ТС 022/2011"]
    with_fish = resolve_regulations(name_aspect, {"fish"})
    assert with_fish == ["ТР ТС 022/2011", "ТР ЕАЭС 040/2016"]
    both = resolve_regulations(name_aspect, {"meat", "fish"})
    assert both[0] == "ТР ТС 022/2011" and len(both) == len(set(both)) == 3


# --- Фильтр-список в ретриве (без API) ---

def test_bm25_filter_accepts_list():
    """BM25 со списком регламентов не пропускает чужие документы."""
    allowed = ["ТР ТС 022/2011", "ТР ТС 029/2012"]
    top = RETRIEVER.bm25_search("пищевая добавка в составе", k=10,
                                regulation=allowed)
    assert top, "фильтр вернул пустую выдачу"
    assert all(RETRIEVER.regulation_of[cid] in allowed for cid, _ in top)
    # Строка работает как раньше (обратная совместимость)
    top_one = RETRIEVER.bm25_search("пищевая добавка", k=5,
                                    regulation="ТР ТС 029/2012")
    assert all(RETRIEVER.regulation_of[cid] == "ТР ТС 029/2012"
               for cid, _ in top_one)


def test_qdrant_matchany_filter_without_api():
    """MatchAny-фильтр Qdrant — на синтетической мини-коллекции (сборка
    полной коллекции из npz долгая, её проверяет смоук с API)."""
    from qdrant_client import QdrantClient
    from qdrant_client import models as qm

    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="t",
        vectors_config=qm.VectorParams(size=4, distance=qm.Distance.COSINE))
    regs = ["ТР ТС 022/2011", "ТР ТС 029/2012", "ТР ЕАЭС 040/2016"]
    client.upsert(collection_name="t", points=[
        qm.PointStruct(id=i, vector=[1.0, float(i), 0.0, 0.0],
                       payload={"regulation_id": regs[i % 3]})
        for i in range(9)])
    allowed = ["ТР ТС 029/2012", "ТР ЕАЭС 040/2016"]
    hits = client.query_points(
        collection_name="t", query=[1.0, 0.0, 0.0, 0.0], limit=9,
        query_filter=RETRIEVER._regulation_filter(allowed)).points
    assert hits and all(h.payload["regulation_id"] in allowed for h in hits)
    # Фильтр одного регламента остаётся MatchValue
    hits_one = client.query_points(
        collection_name="t", query=[1.0, 0.0, 0.0, 0.0], limit=9,
        query_filter=RETRIEVER._regulation_filter("ТР ТС 022/2011")).points
    assert hits_one and all(h.payload["regulation_id"] == "ТР ТС 022/2011"
                            for h in hits_one)


# --- Rewriting: конфиг и кэш (без API) ---

def test_rewrite_disabled_returns_empty():
    cfg = {**CFG, "rewrite": {**CFG["rewrite"], "enabled": False}}
    assert rewrite.rewrite_query("любой запрос", client=None, cfg=cfg) == []


def test_rewrite_cache_hit_needs_no_client():
    """Тёплый кэш отдаёт переформулировки без API и без ключа."""
    os.environ.setdefault("CHEAP_MODEL", "test-model")
    model = os.environ["CHEAP_MODEL"]
    with tempfile.TemporaryDirectory() as tmp:
        cache_file = Path(tmp) / "rewrites.json"
        cfg = {**CFG, "rewrite": {**CFG["rewrite"], "cache": str(cache_file)}}
        prompt = cfg["rewrite"]["prompt"].format(n=cfg["rewrite"]["n"])
        key = rewrite._cache_key(model, prompt, "аллергены", cfg["rewrite"]["n"])
        cache_file.write_text(json.dumps(
            {key: {"query": "аллергены", "model": model,
                   "rewrites": ["компоненты, вызывающие аллергические реакции"]}},
            ensure_ascii=False), encoding="utf-8")
        got = rewrite.rewrite_query("аллергены", client=None, cfg=cfg)
        assert got == ["компоненты, вызывающие аллергические реакции"], got


def test_rewrite_cache_miss_without_client_raises():
    """Промах кэша без клиента — понятная ошибка, а не тихий пропуск."""
    os.environ.setdefault("CHEAP_MODEL", "test-model")
    with tempfile.TemporaryDirectory() as tmp:
        cfg = {**CFG, "rewrite": {**CFG["rewrite"],
                                  "cache": str(Path(tmp) / "none.json")}}
        try:
            rewrite.rewrite_query("нет такого в кэше", client=None, cfg=cfg)
        except RuntimeError:
            return
        raise AssertionError("ожидали RuntimeError при промахе кэша без клиента")


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
