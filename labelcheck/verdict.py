"""Вердикты по аспектам: MAIN_MODEL + обязательные цитаты пунктов регламентов.

Конвейер на один макет (`check_layout`):
1. Категории продукта — CategoryDetector по наименованию и составу из
   vision-JSON (маркер только расширяет поиск, применимость решает вердикт).
2. Для каждого регламентного аспекта:
   - факты с макета: тексты регионов его vision_kinds (пустой список =
     все регионы, кроме technical); регион «требует ручной проверки»
     помечается ненадёжным;
   - контекст норм — BASIS-FIRST (решение Сергея, День 6): ВСЕ тело-пункты
     оснований аспекта идут в промпт детерминированно, без ретрива — нормы,
     которые мы знаем точно, не соревнуются за место в топ-К; ретрив
     добавляет сверху `verdict.retrieval_extra` чанков — строки табличных
     приложений (их не перечислить заранее: какие строки прил.2 029 нужны,
     зависит от Е-кодов конкретного состава — они подставляются в запрос)
     и нормы вне basis;
   - вердикт MAIN_MODEL: {status, applicable, citations, explanation}.
3. Валидация цитат КОДОМ (правило ТЗ §4.3, не обещание модели): цитируемый
   chunk_id обязан быть среди переданных пунктов, quote — дословной
   подстрокой его текста; «соответствует» / «возможное нарушение» без
   валидной цитаты принудительно понижается до «требует ручной проверки».
4. Аспект 17 (шрифт) — всегда «требует ручной проверки» (решение Сергея:
   vision не меряет высоту букв в мм, замер — в BACKLOG). Аспекты 18–19 —
   «прочие замечания»: штрихкод без LLM, орфография CHEAP-моделью.

Все параметры — labelcheck/config.yaml → verdict.
"""

import json
import os
import re
import time

from labelcheck.aspects import CategoryDetector, find_basis_chunks, load_aspects, resolve_regulations
from labelcheck.retrieval import ROOT, load_config, rrf_fuse
from labelcheck.vision import TokenTally

STATUS_COMPLIANT = "соответствует"
STATUS_VIOLATION = "возможное нарушение"
STATUS_MANUAL = "требует ручной проверки"
STATUSES = (STATUS_COMPLIANT, STATUS_VIOLATION, STATUS_MANUAL)

VISION_READ_OK = "прочитано"  # статус региона в layout-JSON (vision.STATUS_OK)

SYSTEM_PROMPT = """Ты — проверяющий маркировку пищевой продукции на соответствие
техническим регламентам ЕАЭС. Тебе дают: аспект проверки, факты с макета
упаковки (дословно прочитанные тексты) и пункты регламентов.

Верни JSON:
{"status": "соответствует" | "возможное нарушение" | "требует ручной проверки",
 "applicable": true | false,
 "citations": [{"chunk_id": "<id пункта из списка>", "quote": "<дословная выдержка из текста этого пункта>"}],
 "explanation": "<2-5 предложений по-русски: что проверил и почему такой вывод>"}

Жёсткие правила:
1. Цитировать можно ТОЛЬКО пункты из переданного списка, quote — дословная
   выдержка из текста пункта (копируй без изменений). Вердикт «соответствует»
   или «возможное нарушение» обязан опираться на цитату.
2. Если требование к данному продукту не применяется (например, аспект об
   импортёре, а признаков импорта нет) — applicable: false, status
   «соответствует», объясни почему не применяется.
3. Неуверенность, нехватка данных, факт из ненадёжно прочитанного региона,
   отсутствие подходящего пункта в списке — status «требует ручной проверки».
   Додуманный вердикт хуже честного «не знаю»: галлюцинация дороже неполноты.
4. Проверяй только формальное соответствие маркировки; достоверность
   заявленных значений по существу не оцениваешь.
5. Тебе передан не весь регламент: ОТСУТСТВИЕ в списке пункта,
   подтверждающего разрешённость или требование, — не доказательство
   нарушения. Не хватает нормы для вывода — «требует ручной проверки»."""


# ── нормализация и извлечение ────────────────────────────────────────────────

_WS = re.compile(r"\s+")
_E_CODE = re.compile(r"(?<![а-яёa-z0-9])[eе]\s?(\d{3,4}[a-dа-г]?)(?![0-9])", re.I)


def norm_text(s: str) -> str:
    """Для проверки «quote — подстрока пункта»: регистр, ё→е, пробелы.
    Модель может сменить регистр или схлопнуть перенос — это не подделка."""
    return _WS.sub(" ", s.lower().replace("ё", "е")).strip()


def extract_e_codes(text: str) -> list[str]:
    """Е-коды из текста в единой форме («е471»): латиница/кириллица, пробел
    после буквы. Для динамического запроса аспектов о добавках."""
    seen = []
    for m in _E_CODE.finditer(text):
        code = "е" + m.group(1).lower()
        if code not in seen:
            seen.append(code)
    return seen


# Аспекты, чей поисковый запрос дополняется фактами макета (Е-коды состава).
DYNAMIC_E_CODE_ASPECTS = {"additives", "warning_labels"}


def ecode_lookup_chunks(codes: list[str], chunks: list[dict],
                        reg: str = "ТР ТС 029/2012",
                        appendix: str = "2", limit: int = 6) -> list[dict]:
    """Детерминированная выборка окон приложения по Е-кодам состава.

    Ретрив теряет нужную строку среди сотен окон прил.2 029 — а по коду её
    можно достать точно: ищем токен кода (оба алфавита, допустим пробел
    «Е 621») в тексте окон приложения. Одно окно может закрывать несколько
    кодов — дубли не берём. Урок сверки Дня 6: у добавки может быть
    несколько классов, строка перечня показывает их все."""
    found, seen = [], set()
    for code in codes:
        digits = code.lstrip("eе")
        pat = re.compile(rf"(?<![0-9а-яёa-z])[eе]\s?{re.escape(digits)}(?![0-9])", re.I)
        for chunk in chunks:
            if chunk["regulation_id"] != reg or str(chunk.get("appendix")) != appendix:
                continue
            if chunk["chunk_id"] not in seen and pat.search(chunk["text"]):
                seen.add(chunk["chunk_id"])
                found.append(chunk)
                break  # первое окно с этим кодом; следующий код
        if len(found) >= limit:
            break
    return found


# ── факты с макета ───────────────────────────────────────────────────────────

def collect_facts(layout: dict) -> list[dict]:
    """ВСЕ прочитанные регионы макета, кроме technical (обвязка дизайнера).

    Каждый аспект получает весь макет (решение Сергея после e2e Дня 6):
    kind-метки обзора недетерминированы, и фильтрация фактов по ним давала
    ложное нарушение (импортёр лежал в регионе состава) и ложные «ручные
    проверки» (датировка без региона dates_storage). vision_kinds аспекта —
    теперь ПОДСКАЗКА промпту, не фильтр. Регион со статусом «требует ручной
    проверки» помечается reliable=False."""
    facts = []
    for region in layout.get("regions", []):
        if region["kind"] == "technical":
            continue
        text = (region.get("text") or "").strip()
        if not text:
            continue
        facts.append({"region": region["id"], "kind": region["kind"],
                      "lang": region.get("lang", "none"),
                      "reliable": region.get("status") == VISION_READ_OK,
                      "text": text})
    return facts


def facts_block(facts: list[dict]) -> str:
    if not facts:
        return "(регионов этого типа на макете не найдено)"
    lines = []
    for f in facts:
        flag = "" if f["reliable"] else " [⚠ прочитан ненадёжно — требует ручной проверки]"
        lines.append(f"— регион {f['region']} ({f['kind']}, язык {f['lang']}){flag}:\n{f['text']}")
    return "\n\n".join(lines)


# ── контекст норм: basis-first ───────────────────────────────────────────────

def chunk_address(chunk: dict) -> str:
    parts = [chunk["regulation_id"]]
    if chunk.get("appendix"):
        parts.append(f"приложение {chunk['appendix']}")
    if chunk.get("subsection"):
        parts.append("ч." + chunk["subsection"].split(".")[0] + "." + chunk["subsection"].split(".")[1])
    elif chunk.get("section"):
        parts.append(chunk["section"].split(".")[0].strip().title()
                     if chunk["section"][:6].upper().startswith("СТАТЬЯ")
                     else "раздел " + chunk["section"].split(".")[0])
    if chunk.get("clause"):
        parts.append(f"п.{chunk['clause']}")
    if chunk.get("part"):
        parts.append(f"(часть {chunk['part']})")
    return ", ".join(parts)


def gather_context(aspect: dict, categories: set[str], chunks: list[dict],
                   search_fn, cfg: dict, facts_text: str = "",
                   scope_basis: list[dict] | None = None) -> tuple[list[dict], list[dict]]:
    """(basis_chunks, retrieval_chunks) для промпта вердикта.

    basis: все ТЕЛО-пункты оснований аспекта в регламентах активных категорий
    + пункты ОБЛАСТИ ПРИМЕНЕНИЯ активных категорийных ТР (scope_basis из
    aspects.yaml: модель видит исключения вроде 034 п.4в «не про птицу» и
    может ставить applicable: false) — целиком, детерминированно; перебор
    max_context_chars самим basis — ошибка конфигурации (громкое исключение,
    не тихое усечение). Дальше, для аспектов о добавках, — строки приложений
    по Е-кодам состава (детерминированный lookup). retrieval: топ
    retrieval_extra слитых выдач по запросам аспекта, без дублей, в остаток
    бюджета.
    """
    vcfg = cfg["verdict"]
    active = set(resolve_regulations(aspect, categories))
    order = {c["chunk_id"]: i for i, c in enumerate(chunks)}

    basis_entries = list(aspect.get("basis", []))
    if scope_basis and aspect.get("category_regulations"):
        basis_entries += [b for b in scope_basis if b["reg"] in active]

    basis_chunks, seen = [], set()
    for basis in basis_entries:
        if "appendix" in basis or basis["reg"] not in active:
            continue
        for chunk in find_basis_chunks(basis, chunks):
            if chunk["chunk_id"] not in seen:
                seen.add(chunk["chunk_id"])
                basis_chunks.append(chunk)
    basis_chunks.sort(key=lambda c: order[c["chunk_id"]])

    budget = vcfg["max_context_chars"]
    basis_chars = sum(len(c["text"]) for c in basis_chunks)
    if basis_chars > budget:
        raise ValueError(
            f"аспект {aspect['id']} ({aspect['key']}): basis {basis_chars} симв. "
            f"> verdict.max_context_chars {budget} — это ошибка конфигурации, "
            "basis не усекается")
    remaining = budget - basis_chars

    codes = extract_e_codes(facts_text) if facts_text else []
    lookup_chunks = []
    if aspect["key"] in DYNAMIC_E_CODE_ASPECTS and codes:
        for chunk in ecode_lookup_chunks(codes, chunks):
            if chunk["chunk_id"] in seen or len(chunk["text"]) > remaining:
                continue
            seen.add(chunk["chunk_id"])
            lookup_chunks.append(chunk)
            remaining -= len(chunk["text"])

    queries = list(aspect.get("queries", []))
    if aspect["key"] in DYNAMIC_E_CODE_ASPECTS and codes:
        queries.append("пищевая добавка " + " ".join(codes))

    rankings = [[cid for cid, _ in search_fn(q, sorted(active))] for q in queries]
    fused = rrf_fuse(rankings, cfg["search"]["rrf_k"], top_k=len(order))

    by_id = {c["chunk_id"]: c for c in chunks}
    extra = []
    for cid, _ in fused:
        if len(extra) == vcfg["retrieval_extra"]:
            break
        if cid in seen:
            continue
        chunk = by_id[cid]
        if len(chunk["text"]) > remaining:
            continue
        extra.append(chunk)
        remaining -= len(chunk["text"])
        seen.add(cid)
    return basis_chunks, lookup_chunks + extra


# ── арифметика пищевой ценности (без LLM; коэффициенты — прил.4 к 022) ──────

_NUTR_NUM = r"(\d+(?:[.,]\d+)?)"


def _find_value(text: str, *labels: str) -> float | None:
    for label in labels:
        m = re.search(label + r"[^\d\n]{0,20}" + _NUTR_NUM, text, re.I)
        if m:
            return float(m.group(1).replace(",", "."))
    return None


def nutrition_arithmetic(facts_text: str) -> dict | None:
    """Сверка заявленных ккал/кДж с расчётом по БЖУ (белки и углеводы —
    4 ккал/г и 17 кДж/г, жиры — 9 ккал/г и 37 кДж/г, прил.4 к 022).
    Возвращает расчёт для промпта вердикта и отчёта; None — если значения
    не распарсились. Спирт/полиолы в расчёт не входят (v1, отметка ниже)."""
    protein = _find_value(facts_text, r"белки")
    fat = _find_value(facts_text, r"жиры")
    carbs = _find_value(facts_text, r"углеводы")
    m = re.search(_NUTR_NUM + r"\s*кДж\s*/\s*" + _NUTR_NUM + r"\s*ккал", facts_text, re.I)
    if m:
        stated_kj, stated_kcal = (float(m.group(1).replace(",", ".")),
                                  float(m.group(2).replace(",", ".")))
    else:
        stated_kcal = _find_value(facts_text, r"ккал")
        stated_kj = _find_value(facts_text, r"кДж")
    if None in (protein, fat, carbs) or (stated_kcal is None and stated_kj is None):
        return None
    calc_kcal = round(protein * 4 + carbs * 4 + fat * 9, 1)
    calc_kj = round(protein * 17 + carbs * 17 + fat * 37, 1)
    result = {"protein_g": protein, "fat_g": fat, "carbs_g": carbs,
              "calc_kcal": calc_kcal, "calc_kj": calc_kj,
              "stated_kcal": stated_kcal, "stated_kj": stated_kj}
    for unit, stated, calc in (("kcal", stated_kcal, calc_kcal),
                               ("kj", stated_kj, calc_kj)):
        if stated:
            result[f"dev_{unit}_pct"] = round(abs(stated - calc) / stated * 100, 1)
    return result


def arithmetic_note(arith: dict | None) -> str:
    if not arith:
        return ""
    parts = [f"белки {arith['protein_g']} г, жиры {arith['fat_g']} г, "
             f"углеводы {arith['carbs_g']} г → расчёт: {arith['calc_kcal']} ккал / "
             f"{arith['calc_kj']} кДж"]
    if arith.get("stated_kcal"):
        parts.append(f"заявлено {arith['stated_kcal']} ккал "
                     f"(отклонение {arith.get('dev_kcal_pct')}%)")
    if arith.get("stated_kj"):
        parts.append(f"заявлено {arith['stated_kj']} кДж "
                     f"(отклонение {arith.get('dev_kj_pct')}%)")
    return ("\nАРИФМЕТИЧЕСКАЯ СВЕРКА (расчёт кодом по коэффициентам прил.4 к 022; "
            "спирт и полиолы не учтены): " + "; ".join(parts))


# ── вердикт одного аспекта ───────────────────────────────────────────────────

def build_user_prompt(aspect: dict, facts: list[dict], basis_chunks: list[dict],
                      extra_chunks: list[dict], categories: set[str]) -> str:
    cat_note = (("По словам-маркерам макет ПОХОЖ на продукцию категорий: "
                 f"{', '.join(sorted(categories))} — это эвристика, применимость "
                 "категорийных регламентов оцени сам по фактам.")
                if categories else
                "Категорийные маркеры (мясо/рыба) в макете не найдены — "
                "применяются только горизонтальные регламенты.")
    kinds = aspect.get("vision_kinds") or []
    hint = (f"Подсказка: обычно эти данные — в регионах типов "
            f"{', '.join(kinds)}, но kind-метки неточны, данные могут "
            "оказаться в регионах других типов — просматривай все факты."
            if kinds else "Аспект касается макета целиком.")
    norm_lines = []
    for chunk in basis_chunks + extra_chunks:
        norm_lines.append(f"[{chunk['chunk_id']}] {chunk_address(chunk)}\n{chunk['text']}")
    return (f"АСПЕКТ {aspect['id']}: {aspect['name']}\n"
            f"Что проверить: {aspect['check'].strip()}\n{cat_note}\n{hint}\n\n"
            f"ФАКТЫ С МАКЕТА (весь макет):\n{facts_block(facts)}\n\n"
            f"ПУНКТЫ РЕГЛАМЕНТОВ (цитировать только их, по chunk_id):\n\n"
            + "\n\n".join(norm_lines))


def validate_verdict(raw: dict, allowed_chunks: list[dict]) -> dict:
    """Валидация ответа модели КОДОМ. Возвращает вердикт с полями
    status/applicable/citations/explanation/downgraded_reason."""
    by_id = {c["chunk_id"]: c for c in allowed_chunks}
    problems = []

    status = raw.get("status")
    if status not in STATUSES:
        problems.append(f"неизвестный статус {status!r}")
        status = STATUS_MANUAL
    applicable = bool(raw.get("applicable", True))

    citations = []
    raw_citations = raw.get("citations") or []
    if not isinstance(raw_citations, list):
        raw_citations = []
        problems.append("citations — не список")
    for cit in raw_citations:
        if not isinstance(cit, dict):
            problems.append(f"цитата не в формате объекта: {cit!r}")
            continue
        cid, quote = cit.get("chunk_id"), (cit.get("quote") or "").strip()
        chunk = by_id.get(cid)
        if chunk is None:
            problems.append(f"цитата ссылается на непереданный пункт {cid!r}")
            continue
        if not quote or norm_text(quote) not in norm_text(chunk["text"]):
            problems.append(f"quote не является дословной выдержкой из {cid}")
            continue
        citations.append({"chunk_id": cid, "address": chunk_address(chunk),
                          "quote": quote})

    downgraded = None
    if status in (STATUS_COMPLIANT, STATUS_VIOLATION) and applicable and not citations:
        downgraded = ("вердикт без валидной цитаты пункта — понижен "
                      "автоматически (ТЗ §4.3)" + ("; " + "; ".join(problems) if problems else ""))
        status = STATUS_MANUAL

    return {"status": status, "applicable": applicable, "citations": citations,
            "explanation": (raw.get("explanation") or "").strip(),
            "downgraded_reason": downgraded,
            "citation_problems": problems or None}


def judge_aspect(aspect: dict, facts: list[dict], categories: set[str],
                 chunks: list[dict], search_fn, client, cfg: dict,
                 tally: TokenTally, scope_basis: list[dict] | None = None) -> dict:
    """Полный вердикт одного регламентного аспекта (facts — весь макет,
    собирается один раз на прогон в check_layout)."""
    base = {"id": aspect["id"], "key": aspect["key"], "name": aspect["name"]}

    if aspect["key"] == "font_size":  # решение Сергея: всегда ручная проверка
        return {**base, "status": STATUS_MANUAL, "applicable": True,
                "citations": [], "citation_problems": None, "downgraded_reason": None,
                "explanation": (
                    "Автоматический замер высоты шрифта не выполняется (в бэклоге). "
                    "Проверь вручную по 022 ч.4.12 п.1: наименование, количество, "
                    "дата изготовления и срок годности — строчные не ниже 2 мм; "
                    "состав, условия хранения, изготовитель, рекомендации, пищевая "
                    "ценность — не ниже 0,8 мм; контраст с фоном; для упаковки "
                    "≤ 10 см² действуют послабления (п.4)."),
                "context": {"basis": 0, "retrieval": 0}}

    facts_text = "\n".join(f["text"] for f in facts)
    basis_chunks, extra_chunks = gather_context(
        aspect, categories, chunks, search_fn, cfg, facts_text, scope_basis)
    allowed = basis_chunks + extra_chunks

    arith = nutrition_arithmetic(facts_text) if aspect["key"] == "nutrition" else None
    user_prompt = build_user_prompt(aspect, facts, basis_chunks, extra_chunks,
                                    categories) + arithmetic_note(arith)

    model = os.environ[cfg["verdict"]["model_env"]]
    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user_prompt}])
    tally.add(model, resp.usage)
    try:
        raw = json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        raw = {"status": STATUS_MANUAL,
               "explanation": "модель вернула не-JSON — ответ отброшен"}

    verdict = validate_verdict(raw, allowed)
    result = {**base, **verdict,
              "context": {"basis": len(basis_chunks), "retrieval": len(extra_chunks)}}
    if arith:
        result["arithmetic"] = arith
    return result


# ── «прочие замечания»: аспекты 18–19 ────────────────────────────────────────

_DIGIT_RUN = re.compile(r"\d[\d\s]{6,}\d")


def barcode_check(layout: dict) -> dict:
    """Аспект 18 без LLM: 13 цифр подряд = кандидат EAN-13 (дубли из
    перехлёстных регионов схлопываются, регионы перечисляются). 8 цифр —
    только если 13-значного нет и с оговоркой: спейс-форма даты («29 08
    2026») неотличима от EAN-8 без контрольной цифры (она в BACKLOG)."""
    runs13, runs8 = {}, {}
    for region in layout.get("regions", []):
        for m in _DIGIT_RUN.finditer(region.get("text") or ""):
            digits = re.sub(r"\D", "", m.group())
            if len(digits) == 13:
                runs13.setdefault(digits, []).append(region["id"])
            elif len(digits) == 8:
                runs8.setdefault(digits, []).append(region["id"])
    if runs13:
        items = [f"найден код {d} (регионы: {', '.join(dict.fromkeys(regs))}); "
                 "контрольная цифра не проверялась (в бэклоге)"
                 for d, regs in runs13.items()]
    elif runs8:
        items = [f"возможен EAN-8: {d} (регионы: {', '.join(dict.fromkeys(regs))}) — "
                 "может оказаться датой или служебным числом, проверь глазами"
                 for d, regs in runs8.items()]
    else:
        items = ["штрихкод (13 цифр подряд) в прочитанных текстах не найден — "
                 "проверь макет глазами: возможно, цифры не распознаны"]
    return {"id": 18, "key": "barcode", "name": "Штрихкод", "items": items}


SPELLING_PROMPT = """Ты — корректор русского текста упаковки пищевой продукции.
Найди орфографические и пунктуационные дефекты: пропущенные/лишние пробелы,
незакрытые скобки и кавычки, пропущенные точки, разнобой в написании терминов
и Е-кодов. НЕ комментируй стиль, содержание, англоязычный текст и
соответствие регламентам; переносы строк внутри регионов — артефакт чтения
макета, не дефект. Верни JSON:
{"remarks": [{"region": "<id региона>",
              "quote": "<дословный фрагмент с дефектом>",
              "what": "<в чём дефект, 3-10 слов>"}]}
Если дефектов нет — {"remarks": []}."""


def layer_word_findings(layout: dict) -> list[str]:
    """Детерминированные находки из текстового СЛОЯ PDF — без LLM.

    unread_layer_words — слова слоя, не найденные в vision-тексте. Урок
    сверки Дня 6: vision молча нормализует дефекты («Hалейте» с латинской H
    прочитан как «Налейте», опечатка «plaease» — как «please»), и дефект
    терялся. Слова с кириллско-латинским миксом — почти наверняка
    гомоглифная опечатка макета; остальные буквенные слова — кандидаты."""
    import unicodedata
    findings = []
    for word in layout.get("unread_layer_words", []) or []:
        if not any(ch.isalpha() for ch in word) or len(word) < 4:
            continue  # числа и короткий шум (коды Pantone и т.п.)
        names = [unicodedata.name(ch, "") for ch in word]
        has_cyr = any("CYRILLIC" in n for n in names)
        has_lat = any("LATIN" in n for n in names)
        if has_cyr and has_lat:
            findings.append(
                f"гомоглифная опечатка в макете: слово «{word}» смешивает "
                "кириллицу и латиницу (vision при чтении молча исправил его — "
                "дефект виден только в текстовом слое PDF)")
        elif word.lower() in ("cmyk", "pantone", "yellow", "magenta", "cyan"):
            continue  # техобвязка дизайнера
        else:
            findings.append(
                f"слово слоя «{word}» не найдено в прочитанном тексте — "
                "возможная опечатка макета или пропуск чтения, проверь глазами")
    return findings


def spelling_check(facts: list[dict], client, cfg: dict,
                   tally: TokenTally, layout: dict | None = None) -> dict:
    """Аспект 19: детерминированные находки из слоя + CHEAP-модель по
    русским блокам макета."""
    layer_items = layer_word_findings(layout or {})
    ru_facts = [f for f in facts if f["lang"] in ("ru", "mixed")]
    base = {"id": 19, "key": "spelling", "name": "Орфография и пунктуация RU"}
    if not ru_facts:
        return {**base, "items": layer_items or ["русских блоков в макете не найдено"]}
    model = os.environ[cfg["verdict"]["cheap_env"]]
    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SPELLING_PROMPT},
                  {"role": "user", "content": facts_block(ru_facts)}])
    tally.add(model, resp.usage)
    try:
        remarks = json.loads(resp.choices[0].message.content).get("remarks", [])
    except json.JSONDecodeError:
        remarks = [{"region": "?", "quote": "", "what": "модель вернула не-JSON"}]
    items = []
    for r in remarks:
        if not isinstance(r, dict):
            continue
        quote = f" — «{r['quote']}»" if r.get("quote") else ""
        items.append(f"{r.get('what', '?')}{quote} (регион {r.get('region', '?')})")
    items = layer_items + items
    return {**base, "items": items or ["дефектов не найдено"]}


# ── конвейер на макет ────────────────────────────────────────────────────────

def detect_categories(layout: dict,
                      detector: CategoryDetector) -> tuple[dict[str, list[str]], str]:
    """Категории продукта → ({категория: [стемы]}, охват сканирования).

    Прицельно — по наименованию и составу (маркетинговые тексты и рецепты
    сервировки дают ложные маркеры). Но kind-метки недетерминированы: если
    регионов этих типов с текстом нет — fallback на весь макет (охват
    "full"), иначе категория молча не определится и категорийные регламенты
    не подтянутся. Ложная категория безопаснее пропущенной: она лишь
    расширяет поиск, применимость решает вердикт."""
    targeted = " ".join(
        (r.get("text") or "") for r in layout.get("regions", [])
        if r["kind"] in ("product_name", "composition"))
    if targeted.strip():
        return detector(targeted), "targeted"
    full = " ".join((r.get("text") or "") for r in layout.get("regions", [])
                    if r["kind"] != "technical")
    return detector(full), "full"


def check_layout(layout: dict, retriever, client, cfg: dict | None = None,
                 aspects_data: dict | None = None, search_fn=None,
                 categories_override: set[str] | None = None) -> dict:
    """Все вердикты по макету → структура отчёта (сериализуемый dict).

    categories_override — решение Сергея («кнопки»): None = автодетект по
    маркерам; set() = только горизонтальные регламенты; {"poultry", ...} =
    заданные категории (в отчёте помечается source=manual).
    search_fn(query, regs) -> [(chunk_id, score)] — подменяется в тестах;
    по умолчанию — гибрид с query rewriting.
    """
    from labelcheck.rewrite import hybrid_search_rewritten  # локально: цикл импортов

    t0 = time.time()
    cfg = cfg or load_config()
    aspects_data = aspects_data or load_aspects()
    tally = TokenTally()
    if search_fn is None:
        def search_fn(query, regs):
            return hybrid_search_rewritten(retriever, query, regulation=regs,
                                           client=client, cfg=cfg)

    detector = CategoryDetector(aspects_data["category_markers"],
                                retriever.tokenizer if retriever else None)
    marker_hits, category_scan = detect_categories(layout, detector)
    if categories_override is not None:
        categories = set(categories_override)
        category_scan = "manual"
        marker_hits = {c: marker_hits.get(c, []) for c in categories}
    else:
        categories = set(marker_hits)
    scope_basis = [b for cat in sorted(categories)
                   for b in aspects_data.get("category_scope_basis", {}).get(cat, [])]
    chunks = retriever.chunks
    facts = collect_facts(layout)  # весь макет, один раз на прогон

    verdicts, other = [], []
    for aspect in aspects_data["aspects"]:
        if aspect["group"] == "regulatory":
            verdicts.append(judge_aspect(aspect, facts, categories, chunks,
                                         search_fn, client, cfg, tally,
                                         scope_basis))
        elif aspect["key"] == "barcode":
            other.append(barcode_check(layout))
        elif aspect["key"] == "spelling":
            other.append(spelling_check(facts, client, cfg, tally, layout))

    manual_regions = [
        {"id": r["id"], "kind": r["kind"], "reason": r.get("status_reason") or ""}
        for r in layout.get("regions", [])
        if r.get("status") == STATUS_MANUAL]

    return {
        "meta": {
            "source_pdf": layout.get("meta", {}).get("source_pdf"),
            "source_sha256": layout.get("meta", {}).get("source_sha256"),
            "categories": marker_hits,
            "category_scan": category_scan,
            "tokens": tally.by_model,
            "seconds": round(time.time() - t0, 1),
        },
        "verdicts": verdicts,
        "other_remarks": other,
        "vision": {
            "missing": layout.get("missing", []),
            "text_layer_coverage": layout.get("text_layer_coverage"),
            "manual_regions": manual_regions,
        },
    }


# ── человекочитаемый отчёт ───────────────────────────────────────────────────

def render_markdown(report: dict) -> str:
    m = report["meta"]
    order = {STATUS_VIOLATION: 0, STATUS_MANUAL: 1, STATUS_COMPLIANT: 2}
    verdicts = sorted(report["verdicts"],
                      key=lambda v: (order[v["status"]], v["id"]))
    counts = {s: sum(1 for v in report["verdicts"]
                     if v["status"] == s and v["applicable"]) for s in STATUSES}
    n_na = sum(1 for v in report["verdicts"] if not v["applicable"])

    lines = [f"# Отчёт LabelCheck: {m.get('source_pdf', '—')}", ""]
    cats = (", ".join(f"{c} (маркеры: {', '.join(st) or 'вручную'})"
                      for c, st in m["categories"].items()) or "не определены")
    scan_note = {"targeted": "", "full": " — детект по всему макету (fallback)",
                 "manual": " — заданы вручную"}.get(m.get("category_scan", ""), "")
    lines += [f"- SHA256 макета: `{m.get('source_sha256', '—')}`",
              f"- Категории продукта: {cats}{scan_note}",
              f"- Итог: 🔴 возможное нарушение — {counts[STATUS_VIOLATION]}, "
              f"🟡 требует ручной проверки — {counts[STATUS_MANUAL]}, "
              f"🟢 соответствует — {counts[STATUS_COMPLIANT]}, "
              f"⚪ не применимо — {n_na}", ""]

    category_regs = {"ТР ТС 034/2013", "ТР ЕАЭС 040/2016", "ТР ЕАЭС 051/2021"}
    icons = {STATUS_VIOLATION: "🔴", STATUS_MANUAL: "🟡", STATUS_COMPLIANT: "🟢"}
    lines.append("## Регламентные вердикты")
    for v in verdicts:
        icon = "⚪" if not v["applicable"] else icons[v["status"]]
        cited = {c["address"].split(",")[0] for c in v["citations"]}
        badges = sorted(cited & category_regs)
        badge = f" [{', '.join(b.split()[-1] for b in badges)}]" if badges else ""
        title = f"{icon} {v['id']}. {v['name']} — {v['status']}{badge}"
        if not v["applicable"]:
            title += " (не применимо)"
        lines += ["", f"### {title}", "", v["explanation"] or "—"]
        if badges:
            only_cat = cited and cited <= category_regs
            lines += ["", "*Опирается на категорийный регламент: "
                          + ", ".join(badges)
                          + (" — ТОЛЬКО на него, без горизонтальных норм"
                             if only_cat else "")
                          + ". Проверь применимость категории к продукту.*"]
        if v.get("arithmetic"):
            a = v["arithmetic"]
            lines += ["", f"*Арифметика (код): по БЖУ {a['calc_kcal']} ккал / "
                          f"{a['calc_kj']} кДж; заявлено "
                          f"{a.get('stated_kcal', '—')} ккал / {a.get('stated_kj', '—')} кДж; "
                          f"отклонение {a.get('dev_kcal_pct', '—')}% / "
                          f"{a.get('dev_kj_pct', '—')}%.*"]
        for c in v["citations"]:
            lines += ["", f"> {c['quote']}", f"> — {c['address']}"]
        if v["downgraded_reason"]:
            lines += ["", f"*Понижено автоматически: {v['downgraded_reason']}*"]

    lines += ["", "## Прочие замечания (не требования ТР)"]
    for block in report["other_remarks"]:
        lines += ["", f"### {block['id']}. {block['name']}", ""]
        lines += [f"- {item}" for item in block["items"]]

    vis = report["vision"]
    lines += ["", "## Vision-предупреждения", ""]
    cov = vis["text_layer_coverage"]
    lines.append("- Покрытие текстового слоя: "
                 + (f"{cov * 100:.0f}%" if isinstance(cov, (int, float))
                    else "текстового слоя в PDF нет"))
    lines.append("- Не найденные обязательные блоки: "
                 + (", ".join(vis["missing"]) or "нет"))
    if vis["manual_regions"]:
        for r in vis["manual_regions"]:
            lines.append(f"- Регион {r['id']} ({r['kind']}) прочитан ненадёжно: "
                         f"{r['reason'] or 'см. layout-JSON'}")
    else:
        lines.append("- Все регионы прочитаны уверенно")

    t = m.get("tokens") or {}
    if t:
        lines += ["", "## Расход токенов", ""]
        for model, d in t.items():
            lines.append(f"- {model}: {d['prompt']}+{d['completion']} "
                         f"({d['calls']} вызовов)")
    return "\n".join(lines) + "\n"
