"""Тесты вердиктов без API: валидация цитат, basis-first контекст, отчёт.

LLM подменяется фейковым клиентом с заготовленными JSON-ответами; поиск —
функцией-заглушкой. Главные проверки: «соответствует»/«возможное нарушение»
без валидной дословной цитаты понижается кодом до «требует ручной проверки»
(ТЗ §4.3), и basis-пункты не теряются и не режутся ни при каких настройках.

Запуск из корня репозитория:  python tests/test_verdict.py
(совместим и с pytest: pytest tests/)
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import json

from labelcheck.aspects import load_aspects
from labelcheck.retrieval import Retriever, load_config
from labelcheck import verdict as V

os.environ.setdefault("MAIN_MODEL", "test-main")
os.environ.setdefault("CHEAP_MODEL", "test-cheap")

CFG = load_config()
DATA = load_aspects()
BY_KEY = {a["key"]: a for a in DATA["aspects"]}
RETRIEVER = Retriever(CFG)  # без API: BM25 + корпус
CHUNKS = RETRIEVER.chunks


# ── фейковый OpenAI-клиент ───────────────────────────────────────────────────

class _Usage:
    prompt_tokens = 10
    completion_tokens = 5


class FakeClient:
    """Отдаёт заготовленные JSON-ответы по кругу; пишет историю вызовов."""

    def __init__(self, payloads):
        self._payloads = payloads
        self.calls = []

        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                payload = outer._payloads[min(len(outer.calls) - 1,
                                              len(outer._payloads) - 1)]

                class _Resp:
                    usage = _Usage()

                    class _Choice:
                        class message:
                            content = json.dumps(payload, ensure_ascii=False)
                    choices = [_Choice()]
                return _Resp()

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def no_search(query, regs):
    return []


# ── нормализация и извлечение ────────────────────────────────────────────────

def test_norm_text_tolerates_case_yo_whitespace():
    """Смена регистра, ё→е и схлопнутый перенос — не подделка цитаты."""
    assert V.norm_text("Пищевая  ПРОДУКЦИЯ\nманкировкЁ") == \
           V.norm_text("пищевая продукция манкировке")


def test_extract_e_codes_both_alphabets_and_space():
    """E322 латиницей, Е 471 с пробелом, е160a — одна форма «е…»."""
    codes = V.extract_e_codes("эмульгатор E322, Е 471 и краситель е160a; вода")
    assert codes == ["е322", "е471", "е160a"], codes


# ── факты с макета ───────────────────────────────────────────────────────────

LAYOUT = {
    "meta": {"source_pdf": "test.pdf", "source_sha256": "0" * 64},
    "regions": [
        {"id": "r1", "kind": "product_name", "lang": "ru",
         "text": "Пельмени со свининой замороженные", "status": "прочитано"},
        {"id": "r2", "kind": "composition", "lang": "ru",
         "text": "Состав: свинина, мука, вода, эмульгатор Е471",
         "status": "прочитано"},
        {"id": "r3", "kind": "marks", "lang": "none",
         "text": "4601234567890", "status": "прочитано"},
        {"id": "r4", "kind": "manufacturer", "lang": "ru",
         "text": "ООО Пример, г. Пример", "status": "требует ручной проверки",
         "status_reason": "стилизованный шрифт"},
        {"id": "r5", "kind": "technical", "lang": "none",
         "text": "PANTONE 485C", "status": "прочитано"},
    ],
    "missing": [], "text_layer_coverage": 0.98,
}


def test_collect_facts_all_but_technical():
    """Каждый аспект видит весь макет: kind-метки — не фильтр (урок e2e:
    импортёр лежал в регионе состава, датировка — без своего региона)."""
    facts = V.collect_facts(LAYOUT)
    assert [f["region"] for f in facts] == ["r1", "r2", "r3", "r4"]


def test_unreliable_region_flagged():
    """Регион «требует ручной проверки» → факт reliable=False и метка в промпте."""
    facts = [f for f in V.collect_facts(LAYOUT) if f["region"] == "r4"]
    assert facts[0]["reliable"] is False
    assert "ненадёжно" in V.facts_block(facts)


def test_vision_kinds_become_prompt_hint():
    """vision_kinds аспекта — подсказка в промпте, не отсечение фактов."""
    facts = V.collect_facts(LAYOUT)
    prompt = V.build_user_prompt(BY_KEY["composition"], facts, [], [], set())
    assert "composition" in prompt and "kind-метки неточны" in prompt
    assert "r4" in prompt  # регион «чужого» типа тоже в фактах


# ── basis-first контекст ─────────────────────────────────────────────────────

def test_basis_body_chunks_all_included():
    """Все тело-пункты активных регламентов в контексте — сколько бы их ни было."""
    aspect = BY_KEY["composition"]  # 10 пунктов 022 + 6 пунктов 034
    basis, extra = V.gather_context(aspect, {"meat"}, CHUNKS, no_search, CFG)
    regs = {c["regulation_id"] for c in basis}
    assert regs == {"ТР ТС 022/2011", "ТР ТС 034/2013"}
    assert len(basis) >= 16, len(basis)
    assert extra == []


def test_basis_respects_categories():
    """Без категорий категорийные пункты (034/040) в контекст не попадают."""
    basis, _ = V.gather_context(BY_KEY["composition"], set(), CHUNKS,
                                no_search, CFG)
    assert {c["regulation_id"] for c in basis} == {"ТР ТС 022/2011"}


def test_basis_overflow_raises_loudly():
    """basis больше бюджета — громкая ошибка конфига, не тихое усечение."""
    cfg = {**CFG, "verdict": {**CFG["verdict"], "max_context_chars": 100}}
    try:
        V.gather_context(BY_KEY["composition"], set(), CHUNKS, no_search, cfg)
    except ValueError as e:
        assert "не усекается" in str(e)
        return
    raise AssertionError("ожидали ValueError при переполнении basis")


def test_retrieval_extra_dedups_and_caps():
    """Ретрив-добавка: без дублей с basis, не больше retrieval_extra штук."""
    aspect = BY_KEY["allergens"]
    basis_only, _ = V.gather_context(aspect, set(), CHUNKS, no_search, CFG)
    basis_ids = [c["chunk_id"] for c in basis_only]
    outside = [c["chunk_id"] for c in CHUNKS
               if c["chunk_id"] not in basis_ids][:10]

    def search(query, regs):
        return [(cid, 1.0) for cid in basis_ids + outside]

    basis, extra = V.gather_context(aspect, set(), CHUNKS, search, CFG)
    extra_ids = [c["chunk_id"] for c in extra]
    assert len(extra) == CFG["verdict"]["retrieval_extra"]
    assert not set(extra_ids) & set(basis_ids)


def test_ecode_lookup_finds_appendix_rows():
    """Е-код-lookup: строка прил.2 029 для е621 достаётся детерминированно,
    со всеми допустимыми классами добавки (урок сверки Дня 6)."""
    rows = V.ecode_lookup_chunks(["е621", "е322"], CHUNKS)
    assert rows, "lookup ничего не нашёл"
    assert all(c["regulation_id"] == "ТР ТС 029/2012" and
               str(c.get("appendix")) == "2" for c in rows)
    joined = " ".join(c["text"] for c in rows).lower()
    assert "глутамат натрия" in joined
    assert "лецитин" in joined  # строка Е322 несёт оба класса


def test_lookup_rows_enter_additives_context():
    """Строки прил.2 по кодам состава попадают в контекст аспекта 4 даже
    при пустом ретриве."""
    _, extra = V.gather_context(BY_KEY["additives"], set(), CHUNKS, no_search,
                                CFG, facts_text="эмульгатор E322")
    assert any("лецитин" in c["text"].lower() for c in extra), \
        [c["chunk_id"] for c in extra]


def test_scope_basis_added_for_active_category():
    """Пункт области применения категорийного ТР попадает в basis вердикта:
    модель видит исключения (034 п.4в — птица не мясо) и может ставить
    applicable: false."""
    from labelcheck.aspects import load_aspects
    scope = load_aspects()["category_scope_basis"]
    flat = [b for cat in ("poultry",) for b in scope[cat]]
    basis, _ = V.gather_context(BY_KEY["product_name"], {"poultry"}, CHUNKS,
                                no_search, CFG, scope_basis=flat)
    texts = " ".join(c["text"] for c in basis)
    assert "не распространяется" in texts, "пункт области применения не в basis"


def test_nutrition_arithmetic():
    """Арифметика КБЖУ (прил.4 к 022): расчёт и отклонения без LLM."""
    text = ("Энергетическая ценность 470 кДж / 111 ккал\n"
            "Жиры 1.6 г\nУглеводы 15 г\nБелки 8.2 г")
    a = V.nutrition_arithmetic(text)
    assert a and a["calc_kcal"] == 107.2 and a["calc_kj"] == 453.6
    assert a["stated_kcal"] == 111 and a["stated_kj"] == 470
    assert 3 <= a["dev_kcal_pct"] <= 4 and 3 <= a["dev_kj_pct"] <= 4
    assert V.nutrition_arithmetic("тут нет цифр") is None


def test_layer_word_findings_homoglyph():
    """Слова слоя, не найденные в vision-тексте, доходят до отчёта;
    кир/лат-микс помечается как гомоглифная опечатка («Hалейте»)."""
    layout = {"unread_layer_words": ["hалейте", "plaease", "cmyk", "230"]}
    items = V.layer_word_findings(layout)
    assert any("гомоглиф" in i and "hалейте" in i for i in items), items
    assert any("plaease" in i for i in items), items
    assert not any("cmyk" in i or "230" in i for i in items), items


def test_categories_override():
    """«Кнопки» категорий: override отключает автодетект (манду без категорий
    вообще, несмотря на курицу в составе)."""
    client = FakeClient([{"status": V.STATUS_MANUAL, "applicable": True,
                          "citations": [], "explanation": "заглушка"}])
    report = V.check_layout(LAYOUT, RETRIEVER, client, CFG, DATA,
                            search_fn=no_search, categories_override=set())
    assert report["meta"]["categories"] == {}
    assert report["meta"]["category_scan"] == "manual"


def test_dynamic_e_code_query_for_additives():
    """Аспект о добавках дополняет запрос Е-кодами из фактов макета."""
    seen_queries = []

    def search(query, regs):
        seen_queries.append(query)
        return []

    V.gather_context(BY_KEY["additives"], set(), CHUNKS, search, CFG,
                     facts_text="состав: эмульгатор E471, краситель е120")
    assert any("е471" in q and "е120" in q for q in seen_queries), seen_queries


# ── валидация вердикта ───────────────────────────────────────────────────────

ALLOWED = [{"chunk_id": "c1", "regulation_id": "ТР ТС 022/2011",
            "subsection": "4.4. Общие требования", "clause": "1",
            "text": "Входящие в состав компоненты указываются в порядке "
                    "убывания их массовой доли."}]


def test_citation_ok_keeps_status():
    v = V.validate_verdict(
        {"status": "возможное нарушение", "applicable": True,
         "citations": [{"chunk_id": "c1",
                        "quote": "в порядке убывания их массовой доли"}],
         "explanation": "x"}, ALLOWED)
    assert v["status"] == "возможное нарушение" and not v["downgraded_reason"]
    assert v["citations"][0]["address"].startswith("ТР ТС 022/2011")


def test_foreign_chunk_id_downgrades():
    """Ссылка на непереданный пункт отбрасывается → статус понижается."""
    v = V.validate_verdict(
        {"status": "соответствует",
         "citations": [{"chunk_id": "tr_ts_022_2011:9999", "quote": "что-то"}]},
        ALLOWED)
    assert v["status"] == V.STATUS_MANUAL and v["downgraded_reason"]


def test_rewritten_quote_downgrades():
    """Пересказанная (не дословная) цитата — не цитата."""
    v = V.validate_verdict(
        {"status": "соответствует",
         "citations": [{"chunk_id": "c1",
                        "quote": "компоненты перечисляют по убыванию долей"}]},
        ALLOWED)
    assert v["status"] == V.STATUS_MANUAL and v["downgraded_reason"]


def test_no_citations_downgrades():
    v = V.validate_verdict({"status": "соответствует", "citations": []}, ALLOWED)
    assert v["status"] == V.STATUS_MANUAL


def test_not_applicable_needs_no_citation():
    """applicable: false — цитата не требуется, статус не понижается."""
    v = V.validate_verdict(
        {"status": "соответствует", "applicable": False,
         "explanation": "признаков импорта нет"}, ALLOWED)
    assert v["status"] == "соответствует" and not v["downgraded_reason"]


def test_unknown_status_becomes_manual():
    v = V.validate_verdict({"status": "норм"}, ALLOWED)
    assert v["status"] == V.STATUS_MANUAL


def test_manual_without_citations_not_downgraded():
    """«Требует ручной проверки» без цитат — легитимный честный ответ."""
    v = V.validate_verdict({"status": V.STATUS_MANUAL}, ALLOWED)
    assert v["status"] == V.STATUS_MANUAL and not v["downgraded_reason"]


# ── специальные аспекты ──────────────────────────────────────────────────────

def test_font_size_always_manual_no_llm():
    """Аспект 17: всегда ручная проверка, LLM не вызывается (client=None)."""
    v = V.judge_aspect(BY_KEY["font_size"], V.collect_facts(LAYOUT), set(),
                       CHUNKS, no_search, None, CFG, V.TokenTally())
    assert v["status"] == V.STATUS_MANUAL
    assert "2 мм" in v["explanation"] and "0,8 мм" in v["explanation"]


def test_barcode_found_and_deduplicated():
    """Один код в трёх перехлёстных регионах — одно замечание, регионы вместе."""
    layout = {"regions": [
        {"id": "a", "kind": "marks", "text": "4601234567890"},
        {"id": "b", "kind": "other_text", "text": "код 4 601234 567890 внизу"},
        {"id": "c", "kind": "marks", "text": "4601234567890"}]}
    block = V.barcode_check(layout)
    assert len(block["items"]) == 1
    assert "a, b, c" in block["items"][0]


def test_barcode_8_digits_only_with_caveat():
    """8 цифр без 13-значного кода — только с оговоркой (может быть датой),
    а при наличии EAN-13 восьмёрки вообще не показываются."""
    layout = {"regions": [{"id": "a", "kind": "marks", "text": "29 08 2026"}]}
    block = V.barcode_check(layout)
    assert "может оказаться датой" in block["items"][0]
    both = {"regions": [{"id": "a", "kind": "marks",
                         "text": "4601234567890 и 29 08 2026"}]}
    items = V.barcode_check(both)["items"]
    assert len(items) == 1 and "4601234567890" in items[0]


def test_category_fallback_when_kinds_missing():
    """Состав размечен как other_text → прицельный скан пуст, fallback
    на весь макет всё равно находит категорию."""
    layout = {"regions": [{"id": "x", "kind": "other_text", "lang": "ru",
                           "text": "Состав: свинина, мука",
                           "status": "прочитано"}]}
    detector = V.CategoryDetector(DATA["category_markers"], RETRIEVER.tokenizer)
    hits, scan = V.detect_categories(layout, detector)
    assert set(hits) == {"meat"} and scan == "full"
    hits2, scan2 = V.detect_categories(LAYOUT, detector)
    assert set(hits2) == {"meat"} and scan2 == "targeted"


def test_citation_non_dict_ignored_not_crashing():
    """Цитата-строка вместо объекта не роняет валидатор."""
    v = V.validate_verdict({"status": "соответствует",
                            "citations": ["просто строка"]}, ALLOWED)
    assert v["status"] == V.STATUS_MANUAL and v["downgraded_reason"]


def test_spelling_item_format():
    """Орфография: «что не так» и цитата — раздельные поля."""
    client = FakeClient([{"remarks": [{"region": "r2", "quote": "мука,вода",
                                       "what": "пропущен пробел после запятой"}]}])
    block = V.spelling_check(V.collect_facts(LAYOUT), client, CFG, V.TokenTally())
    assert block["items"] == [
        "пропущен пробел после запятой — «мука,вода» (регион r2)"]


# ── конвейер и отчёт ─────────────────────────────────────────────────────────

def test_check_layout_report_shape():
    """Полный прогон с фейковым LLM: 19 вердиктов, 2 «прочих», категория meat."""
    client = FakeClient([{"status": V.STATUS_MANUAL, "applicable": True,
                          "citations": [], "explanation": "заглушка"}])
    report = V.check_layout(LAYOUT, RETRIEVER, client, CFG, DATA,
                            search_fn=no_search)
    assert len(report["verdicts"]) == 19
    assert [b["id"] for b in report["other_remarks"]] == [18, 19]
    assert set(report["meta"]["categories"]) == {"meat"}  # свинина в составе
    # 18 вызовов MAIN (17-й без LLM) + 1 CHEAP (орфография)
    assert len(client.calls) == 19
    assert report["vision"]["manual_regions"][0]["id"] == "r4"


def test_render_markdown_sections():
    client = FakeClient([{"status": V.STATUS_MANUAL, "applicable": True,
                          "citations": [], "explanation": "заглушка"}])
    report = V.check_layout(LAYOUT, RETRIEVER, client, CFG, DATA,
                            search_fn=no_search)
    md = V.render_markdown(report)
    for needle in ("# Отчёт LabelCheck", "## Регламентные вердикты",
                   "## Прочие замечания", "## Vision-предупреждения",
                   "Категории продукта"):
        assert needle in md, needle


def test_manual_status_requires_next_action_rule():
    """Промпт требует: «ручная проверка» завершается конкретным действием."""
    for needle in ("конкретным следующим действием", "вопрос задать производителю"):
        assert needle in V.SYSTEM_PROMPT, needle


def test_prompt_requires_self_sufficient_explanation():
    """Промпт требует самодостаточности: опираешься на норму — процитируй
    (урок судьи Дня 8: объяснения ссылались на непроцитированные basis-нормы)."""
    for needle in ("опираешься на норму — процитируй",
                   "Читатель отчёта видит только citations"):
        assert needle in V.SYSTEM_PROMPT, needle


def test_ean13_checksum_and_near_misses():
    """Чек-сумма EAN-13: валидный код проходит, битый и 14-значный — ручная
    (кейс прогона v3: штрихкод mango прочитан слитно с лишней цифрой)."""
    assert V.ean13_checksum_ok("4607070255215")        # реальный код mango
    assert V.ean13_checksum_ok("8805957025951")        # реальный код mandu
    assert not V.ean13_checksum_ok("4607070255214")    # битая контрольная
    layout = {"regions": [
        {"id": "a", "kind": "marks", "text": "штрихкод: 4 607070 255215"},
        {"id": "b", "kind": "marks", "text": "41607070255215"},
        {"id": "c", "kind": "marks", "text": "8 805957 025952"}]}
    items = V.barcode_check(layout)["items"]
    joined = " || ".join(items)
    assert "сходится" in joined                        # валидный найден
    assert "НЕ сходится" in joined                     # битый пойман
    assert "14 цифр" in joined                         # слитный пойман


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
