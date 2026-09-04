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

«День надёжности» (эта сессия):
5. Голосование для нестабильных аспектов (verdict.vote_aspect_ids):
   verdict.votes вызовов, большинство валидированных статусов побеждает;
   все статусы разные → «требует ручной проверки».
6. Кэш сырых ответов модели data/verdict_cache.json (НЕ в git): ключ от
   модели + обоих промптов, правка промпта инвалидирует кэш сама.
7. Валидатор самодостаточности КОДОМ: номер пункта в объяснении обязан
   встречаться в адресе или тексте процитированного чанка, иначе
   «соответствует»/«нарушение» понижается до ручной проверки.

Все параметры — labelcheck/config.yaml → verdict.
"""

import hashlib
import json
import os
import re
import time
from collections import Counter

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
   или «возможное нарушение» обязан опираться на цитату. Объяснение должно
   быть самодостаточным: опираешься на норму — процитируй её (добавь в
   citations); не упоминай в explanation пункты и требования, которых нет
   среди процитированных. Читатель отчёта видит только citations.
2. Если требование к данному продукту не применяется (например, аспект об
   импортёре, а признаков импорта нет) — applicable: false, status
   «соответствует», объясни почему не применяется.
3. Неуверенность, нехватка данных, отсутствие подходящего пункта в списке —
   status «требует ручной проверки». Додуманный вердикт хуже честного «не
   знаю»: галлюцинация дороже неполноты. Пометка региона «прочитан
   ненадёжно» сама по себе НЕ основание для ручной проверки (решение
   владельца): делай вывод по прочитанному тексту как есть. ТОЛЬКО если
   среди использованных фактов есть регион с такой пометкой, добавь в
   explanation отдельное предложение с ЕГО реальным id: «регион <id>
   прочитан ненадёжно — вывод требует сверки с макетом»; если помеченных
   регионов нет — ничего об этом не пиши. Ручная проверка уместна, только
   если сам текст такого региона искажён (обрывки, [неразборчиво]) и вывод
   по нему сделать нельзя.
   При статусе «требует ручной проверки» explanation обязан заканчиваться
   конкретным следующим действием: что именно проверить вручную или какой
   вопрос задать производителю/поставщику (например: «запросить вид куриного
   сырья: бескостное мясо или мехобвалка»). Голое «подтвердить нельзя» —
   недостаточно.
4. Проверяй только формальное соответствие маркировки; достоверность
   заявленных значений по существу не оцениваешь. При статусе «возможное
   нарушение» объяснение обязано заканчиваться одним-двумя простыми
   предложениями: ЧТО именно на макете не так, КАК должно быть по
   процитированной норме и что исправить — читатель отчёта не обязан
   реконструировать вывод из рассуждений. Не перечисляй проверки, по
   которым замечаний нет, и не рассуждай о правилах, которые к продукту не
   относятся (например, о виде сырья по птице у продукта без птицы): в
   explanation — только то, что влияет на вывод.
5. Тебе передан не весь регламент: ОТСУТСТВИЕ в списке пункта,
   подтверждающего разрешённость или требование, — не доказательство
   нарушения. Не хватает нормы для вывода — «требует ручной проверки».
6. Слова из иноязычных блоков (корейский, английский и др.) упоминай только
   с русским переводом в скобках и указанием региона: «물엿 (крахмальная
   патока, регион t2r4)». Читатель отчёта не обязан знать эти языки."""


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
# Аспекты, получающие вторым шагом окна ГИГИЕНИЧЕСКИХ нормативов 029
# (прил.3–18: «добавка | пищевая продукция | максимальный уровень») —
# только «Пищевые добавки»: для предупредительных надписей эти таблицы
# не нужны и раздували бы контекст. (Пункт B, день 9.)
HYGIENE_E_CODE_ASPECTS = {"additives"}


def _ecode_pattern(code: str) -> re.Pattern:
    """Регулярка токена Е-кода в тексте: оба алфавита, допустим пробел
    («Е 621»); «е160» не матчит «е1600»."""
    digits = code.lstrip("eе")
    return re.compile(rf"(?<![0-9а-яёa-z])[eе]\s?{re.escape(digits)}(?![0-9])",
                      re.I)


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
        pat = _ecode_pattern(code)
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


def hygiene_lookup_chunks(codes: list[str], chunks: list[dict],
                          cfg: dict) -> list[dict]:
    """Окна ГИГИЕНИЧЕСКИХ приложений 029 (прил.3–18) с Е-кодами состава —
    второй шаг lookup'а (пункт B, день 9): таблицы «добавка | пищевая
    продукция | максимальный уровень» дают вердикту основание сказать
    «разрешён для категории с пределом…» или что категории продукта в
    перечне не видно. Приложения и потолки — config → verdict.ecode_lookup
    (BACKLOG предполагал прил.19/26/12 — по корпусу это ароматизаторы,
    ферменты и носители, Е-кодов там нет; нормативы — в прил.3–18)."""
    el = cfg["verdict"]["ecode_lookup"]
    reg = el["reg"]
    apps = {str(a) for a in el["hygiene_appendices"]}
    per_code, total = el["hygiene_per_code"], el["hygiene_total"]
    found, seen = [], set()
    for code in codes:
        pat = _ecode_pattern(code)
        n = 0
        for chunk in chunks:
            if chunk["regulation_id"] != reg:
                continue
            if str(chunk.get("appendix")) not in apps:
                continue
            if chunk["chunk_id"] in seen or not pat.search(chunk["text"]):
                continue
            seen.add(chunk["chunk_id"])
            found.append(chunk)
            n += 1
            if n >= per_code or len(found) >= total:
                break
        if len(found) >= total:
            break
    return found


# ── кэш ответов вердикт-модели («день надёжности») ───────────────────────────
# В промпты попадают тексты реальных макетов, поэтому файл кэша НЕ в git
# (config → verdict.cache, путь в .gitignore). Ключ включает модель и ОБА
# промпта: правка SYSTEM_PROMPT или сборки user_prompt инвалидирует кэш сама,
# без ручного сброса. salt различает повторные вызовы при голосовании — иначе
# три вызова с одним промптом вернули бы один закэшированный ответ и
# голосование стало бы фикцией. Замеры стабильности гонять ТОЛЬКО с
# --no-cache: кэш даёт фиктивные 100%.


def _verdict_cache_path(cfg: dict):
    return ROOT / cfg["verdict"]["cache"]


def _verdict_cache_key(model: str, user_prompt: str, salt: str = "") -> str:
    raw = "\x1f".join([model, SYSTEM_PROMPT, user_prompt, salt])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _load_json_cache(path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_json_cache(path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=1, sort_keys=True),
                   encoding="utf-8")
    tmp.replace(path)  # атомарная замена: не портим кэш при падении на записи


def call_model(client, model: str, user_prompt: str, cfg: dict,
               tally: TokenTally, use_cache: bool, salt: str = "",
               aspect_key: str = "") -> dict:
    """Один вызов вердикт-модели → сырой dict ответа; с кэшем сырых ответов.

    Кэшируется только валидный JSON — битый ответ не должен залипнуть.
    Попадание в кэш не тратит токены (в tally не пишется)."""
    path = _verdict_cache_path(cfg)
    key = _verdict_cache_key(model, user_prompt, salt)
    if use_cache:
        cached = _load_json_cache(path).get(key)
        if cached is not None:
            return cached["raw"]

    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user_prompt}])
    tally.add(model, resp.usage)
    try:
        raw = json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        return {"status": STATUS_MANUAL,
                "explanation": "модель вернула не-JSON — ответ отброшен"}

    if use_cache:
        cache = _load_json_cache(path)  # перечитать: файл мог пополниться
        cache[key] = {"model": model, "aspect": aspect_key, "salt": salt,
                      "raw": raw}
        _save_json_cache(path, cache)
    return raw


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
        flag = "" if f["reliable"] else " [⚠ прочитан ненадёжно — сверить с макетом]"
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
        if basis["reg"] not in active:
            continue
        # Приложения в basis по умолчанию НЕ грузятся (их строки достаёт
        # ретрив/lookup — прил.2 029 это сотни окон). Флаг always: true
        # кладёт приложение в контекст целиком, как тело-пункты — для
        # компактных перечней вроде прил.1 к 022 (пункт B, день 9).
        if "appendix" in basis and not basis.get("always"):
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
        el = vcfg["ecode_lookup"]
        pool = ecode_lookup_chunks(codes, chunks, reg=el["reg"],
                                   limit=el["appendix_2_limit"])
        if aspect["key"] in HYGIENE_E_CODE_ASPECTS:
            pool += hygiene_lookup_chunks(codes, chunks, cfg)
        for chunk in pool:
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
_KCAL_PER_KJ = 4.184  # 1 ккал = 4,184 кДж (для сверки согласованности записи)


def _num(s: str) -> float:
    return float(s.replace(",", "."))


def _label_values(text: str, labels: dict[str, str]) -> dict[str, float]:
    """Значения нутриентов; направление записи определяется по ВСЕМУ тексту.

    Макеты пишут в двух порядках: «белки 1 г» и «1 г белки» (кейс mango,
    день 9). Пометка направления по каждой метке отдельно даёт смещение
    («белки» цепляют число следующего нутриента), поэтому сначала пробуем
    оба направления целиком и выбираем то, которое нашло больше нутриентов:
    порядок записи на одном макете единый."""
    other = r"(?:белк|жир|углевод|ккал|кдж|кало)"
    num = r"(?P<num>\d+(?:[.,]\d+)?)"
    found = ({}, {})  # 0 — «метка → число», 1 — «число → метка»
    for key, label in labels.items():
        pats = (label + r"(?P<gap>[^\d\n]{0,12})" + num,
                num + r"(?P<gap>\s*(?:г|g)?[^\d\n]{0,6}?)" + label)
        for i, pat in enumerate(pats):
            for m in re.finditer(pat, text, re.I):
                if re.search(other, m.group("gap"), re.I):
                    continue  # между числом и меткой другой нутриент — не наше
                found[i].setdefault(key, _num(m.group("num")))
                break
    forward, backward = found
    return backward if len(backward) > len(forward) else forward


def _find_value(text: str, label: str) -> float | None:
    """Одно значение по метке (энергия и одиночные показатели)."""
    return _label_values(text, {"v": label}).get("v")


def _find_energy(text: str) -> tuple[float | None, float | None]:
    """Заявленные ккал и кДж. Поддержаны форматы: «268 кДж / 64 ккал»,
    «64 ккал / 268 кДж», компактная запись макета «64/268 ккал кДж»
    (значения через дробь, единицы следом — день 9, кейс mango) и
    одиночные упоминания. Пара всегда сверяется по 4,184: если числа
    не согласуются, порядок пробуем поменять местами."""
    pair = re.search(_NUTR_NUM + r"\s*/\s*" + _NUTR_NUM +
                     r"\s*(ккал\s*[/,]?\s*кдж|кдж\s*[/,]?\s*ккал)",
                     text, re.I)
    if pair:
        a, b, units = _num(pair.group(1)), _num(pair.group(2)), pair.group(3).lower()
        return (a, b) if units.startswith("ккал") else (b, a)
    for first, second, order in ((r"кДж", r"ккал", "kj"), (r"ккал", r"кДж", "kcal")):
        m = re.search(_NUTR_NUM + r"\s*" + first + r"\s*[/,]?\s*" +
                      _NUTR_NUM + r"\s*" + second, text, re.I)
        if m:
            x, y = _num(m.group(1)), _num(m.group(2))
            return (y, x) if order == "kj" else (x, y)
    kcal = _find_value(text, r"ккал")
    kj = _find_value(text, r"кДж")
    if kcal and kj and abs(kj / kcal - _KCAL_PER_KJ) > 0.6 and \
            abs(kcal / kj - _KCAL_PER_KJ) < 0.6:
        kcal, kj = kj, kcal  # значения явно перепутаны местами при чтении
    return kcal, kj


def nutrition_arithmetic(facts_text: str) -> dict | None:
    """Сверка заявленных ккал/кДж с расчётом по БЖУ (белки и углеводы —
    4 ккал/г и 17 кДж/г, жиры — 9 ккал/г и 37 кДж/г, прил.4 к 022).
    Возвращает расчёт для промпта вердикта и отчёта; None — если значения
    не распарсились. Спирт/полиолы в расчёт не входят (v1, отметка ниже)."""
    vals = _label_values(facts_text, {"protein": r"белк\w*", "fat": r"жир\w*",
                                      "carbs": r"углевод\w*"})
    protein, fat, carbs = vals.get("protein"), vals.get("fat"), vals.get("carbs")
    stated_kcal, stated_kj = _find_energy(facts_text)
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


def language_note(facts: list[dict]) -> str:
    """Детерминированная сводка языков для аспекта 16 (день 9, решение
    Сергея): аспект качался, потому что модель каждый раз заново решала,
    какие блоки есть по-русски. Считаем кодом по меткам lang регионов —
    модель рассуждает от готовой сводки, а не собирает её заново."""
    if not facts:
        return ""
    by_lang = Counter(f["lang"] for f in facts)
    ru_kinds = sorted({f["kind"] for f in facts if f["lang"] in ("ru", "mixed")})
    non_ru = sorted({f["kind"] for f in facts} - set(ru_kinds))
    langs = ", ".join(f"{k}: {v}" for k, v in sorted(by_lang.items()))
    return ("\nСВОДКА ЯЗЫКОВ (посчитано кодом по регионам макета): "
            f"регионов по языкам — {langs}. "
            f"Типы блоков, где есть русский текст (ru/mixed): "
            f"{', '.join(ru_kinds) or '—'}. "
            f"Типы блоков БЕЗ русского текста: {', '.join(non_ru) or '—'}. "
            "Метки языков автоматические: спорные блоки перепроверь по самим "
            "фактам.")


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


# Ссылка на пункт/часть/статью/приложение в тексте объяснения: префикс
# (п., пп., пункт…, ч., част…, ст., стать…, прил…) + номер. Голое число без
# префикса («в 100 г», «около 6 минут») ссылкой НЕ считается — иначе ложные
# понижения. Числа в перечислении после «и»/запятой («пп. 13 и 14» → только
# 13) сознательно не ловим: пропущенная ссылка не понижает вердикт зря.
_CLAUSE_REF = re.compile(
    r"(?<![а-яёa-z0-9])(?:пп?|пункт[а-яё]*|ч|част[а-яё]*|ст|стать[а-яё]*|"
    r"прил[а-яё]*)\.?\s*(\d+(?:\.\d+)*)", re.I)

_NUM_TOKEN = re.compile(r"\d+(?:\.\d+)*")


def unresolved_clause_refs(explanation: str, citations: list[dict],
                           by_id: dict) -> list[str]:
    """Номера пунктов из объяснения, которых нет ни в адресе, ни в тексте
    ни одного ПРОЦИТИРОВАННОГО чанка («день надёжности», урок судьи Дня 8:
    объяснения ссылались на нормы, которых читатель отчёта не видит).
    Сравнение — по целым числовым токенам: «14» не зачтётся за «4.14»."""
    refs = {m.group(1) for m in _CLAUSE_REF.finditer(explanation or "")}
    if not refs:
        return []
    hay = " ".join(
        chunk_address(by_id[c["chunk_id"]]) + " " + by_id[c["chunk_id"]]["text"]
        for c in citations if c["chunk_id"] in by_id)
    have = set(_NUM_TOKEN.findall(hay))
    return sorted(refs - have)


# Число с единицей °С или % в объяснении. Только эти две единицы (решение
# Сергея, день 9, кейс «минус 12°С» из ниоткуда в аспекте 9): мм не нужен
# (шрифт всегда ручная проверка), граммы дают ложняки — арифметика КБЖУ
# легально порождает новые числа в граммах.
_UNIT_NUM = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:°\s*[cс]|%)", re.I)


def unresolved_unit_numbers(explanation: str, citations: list[dict],
                            by_id: dict, extra_text: str = "") -> list[str]:
    """Числа с °С / % из объяснения, которых нет ни в текстах процитированных
    пунктов, ни в extra_text (факты макета + арифметическая сверка).
    Запятая-разделитель нормализуется к точке с обеих сторон."""
    refs = {m.group(1).replace(",", ".")
            for m in _UNIT_NUM.finditer(explanation or "")}
    if not refs:
        return []
    hay = " ".join([by_id[c["chunk_id"]]["text"]
                    for c in citations if c["chunk_id"] in by_id]
                   + [extra_text or ""])
    hay = re.sub(r"(?<=\d),(?=\d)", ".", hay)
    have = set(re.findall(r"\d+(?:\.\d+)?", hay))
    return sorted(refs - have)


def validate_verdict(raw: dict, allowed_chunks: list[dict],
                     facts_text: str = "") -> dict:
    """Валидация ответа модели КОДОМ. Возвращает вердикт с полями
    status/applicable/citations/explanation/downgraded_reason.
    facts_text — факты макета (+ арифметическая сверка): числа с °С / %
    из объяснения сверяются и с ними, не только с цитатами."""
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

    # Самодостаточность объяснения («день надёжности»): упомянул номер
    # пункта или число с °С / % — они обязаны быть среди процитированного
    # (числа — либо в фактах макета), иначе читатель отчёта не может
    # проверить вывод.
    if status in (STATUS_COMPLIANT, STATUS_VIOLATION) and applicable:
        expl = raw.get("explanation") or ""
        missing_refs = unresolved_clause_refs(expl, citations, by_id)
        missing_units = unresolved_unit_numbers(expl, citations, by_id,
                                                facts_text)
        if missing_refs or missing_units:
            parts = []
            if missing_refs:
                parts.append(f"пункты ({', '.join(missing_refs)})")
            if missing_units:
                parts.append(f"значения с °С/% ({', '.join(missing_units)})")
            downgraded = ("объяснение опирается на " + " и ".join(parts) +
                          ", которых нет среди процитированных норм и фактов "
                          "макета — понижено автоматически (самодостаточность)")
            status = STATUS_MANUAL

    return {"status": status, "applicable": applicable, "citations": citations,
            "explanation": (raw.get("explanation") or "").strip(),
            "downgraded_reason": downgraded,
            "citation_problems": problems or None}


def judge_aspect(aspect: dict, facts: list[dict], categories: set[str],
                 chunks: list[dict], search_fn, client, cfg: dict,
                 tally: TokenTally, scope_basis: list[dict] | None = None,
                 use_cache: bool = False) -> dict:
    """Полный вердикт одного регламентного аспекта (facts — весь макет,
    собирается один раз на прогон в check_layout).

    Голосование («день надёжности»): для аспектов из verdict.vote_aspect_ids
    (исторически нестабильные по замеру Дня 8) — verdict.votes вызовов;
    побеждает большинство ВАЛИДИРОВАННЫХ статусов, представитель — первый
    вердикт победившего статуса, статусы всех голосов — в поле "votes".
    Все статусы разные → «требует ручной проверки»: модель нестабильна на
    этом аспекте, решает человек. Остальные аспекты — один вызов."""
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
    user_prompt = (build_user_prompt(aspect, facts, basis_chunks, extra_chunks,
                                     categories)
                   + arithmetic_note(arith)
                   + (language_note(facts) if aspect["key"] == "language" else ""))
    # Числа с °С / % из объяснения сверяются с фактами макета и арифметикой
    # (числа отклонений КБЖУ легитимны — они посчитаны кодом и есть в промпте).
    facts_ctx = facts_text + arithmetic_note(arith)

    vcfg = cfg["verdict"]
    model = os.environ[vcfg["model_env"]]
    n_votes = (max(1, int(vcfg.get("votes", 1)))
               if aspect["id"] in set(vcfg.get("vote_aspect_ids", []))
               else 1)

    votes = []
    for i in range(n_votes):
        raw = call_model(client, model, user_prompt, cfg, tally, use_cache,
                         salt=f"vote{i}" if i else "", aspect_key=aspect["key"])
        votes.append(validate_verdict(raw, allowed, facts_ctx))

    verdict = votes[0]
    if n_votes > 1:
        statuses = [v["status"] for v in votes]
        top_status, top_n = Counter(statuses).most_common(1)[0]
        if top_n > 1:  # есть большинство — его представитель идёт в отчёт
            verdict = next(v for v in votes if v["status"] == top_status)
        else:  # все статусы разные — модель нестабильна, решает человек
            verdict = dict(votes[0])
            verdict["status"] = STATUS_MANUAL
            verdict["downgraded_reason"] = (
                f"голосование: {n_votes} вызова дали {n_votes} разных статуса "
                f"({', '.join(statuses)}) — модель нестабильна на этом "
                "аспекте, нужен взгляд человека")
        verdict = {**verdict, "votes": statuses}

    result = {**base, **verdict,
              "context": {"basis": len(basis_chunks), "retrieval": len(extra_chunks)}}
    if arith:
        result["arithmetic"] = arith
    return result


# ── «прочие замечания»: аспекты 18–19 ────────────────────────────────────────

# Цифры с пробелами внутри строки: «4 601234 567890». Перенос строки НЕ
# входит (REVIEW-LOG R-38: «82\n\n8 805957 025951» склеивалось в 15 цифр, и
# прочитанный штрихкод считался ненайденным).
_DIGIT_RUN = re.compile(r"\d[\d \u00a0\u2009]{6,}\d")


def ean13_checksum_ok(digits: str) -> bool:
    """Контрольная цифра EAN-13: нечётные позиции ×1, чётные ×3 (поднято из
    BACKLOG после прогона v3: штрихкод mango прочитался с лишней цифрой —
    чек-сумма ловит такое кодом, без LLM)."""
    if len(digits) != 13 or not digits.isdigit():
        return False
    total = sum(int(c) * (1 if i % 2 == 0 else 3)
                for i, c in enumerate(digits[:12]))
    return (10 - total % 10) % 10 == int(digits[12])


def barcode_check(layout: dict) -> dict:
    """Аспект 18 без LLM: 13 цифр подряд = кандидат EAN-13 с проверкой
    контрольной цифры (дубли из перехлёстных регионов схлопываются, регионы
    перечисляются). 12/14 цифр — «похоже на битый штрихкод», ручная проверка:
    так в прогоне v3 mango код прочитался слитно с лишней цифрой. 8 цифр —
    только если 13-значного нет и с оговоркой: спейс-форма даты («29 08
    2026») неотличима от EAN-8 без контрольной цифры."""
    runs13, runs8, near = {}, {}, {}
    for region in layout.get("regions", []):
        for m in _DIGIT_RUN.finditer(region.get("text") or ""):
            digits = re.sub(r"\D", "", m.group())
            if len(digits) == 13:
                runs13.setdefault(digits, []).append(region["id"])
            elif len(digits) in (12, 14):
                near.setdefault(digits, []).append(region["id"])
            elif len(digits) == 8:
                runs8.setdefault(digits, []).append(region["id"])
    # Сверка со СЛОЕМ (REVIEW-LOG R-16): выдуманный код 4820140240955 на
    # манду имел верную контрольную цифру — арифметика его не ловит, а слой
    # знает только 8805957025951. Если в слое длинных чисел нет (штрихкод
    # картинкой), сверка невозможна — говорим об этом честно.
    layer_runs = layout.get("layer_digit_runs")   # None — старый layout без поля

    def layer_note(d):
        if layer_runs is None:
            return ""
        if not layer_runs:
            return " (в текстовом слое PDF цифр штрихкода нет — сверить не с чем)"
        if any(d in run for run in layer_runs):
            return "; подтверждён текстовым слоем PDF"
        return (" — в текстовом слое PDF такого кода НЕТ: вероятная ошибка "
                "чтения (модель могла додумать цифры), требуется ручная проверка")

    items = []
    for d, regs in runs13.items():
        where = ", ".join(dict.fromkeys(regs))
        if ean13_checksum_ok(d):
            items.append(f"найден код {d} (регионы: {where}); "
                         "контрольная цифра EAN-13 сходится" + layer_note(d))
        else:
            items.append(f"найден код {d} (регионы: {where}), но контрольная "
                         "цифра EAN-13 НЕ сходится — код неверно прочитан "
                         "или битый, требуется ручная проверка" + layer_note(d))
    for d in layer_runs or []:
        # код есть в слое, но ни в одном прочитанном тексте — vision его
        # пропустил или прочитал с ошибкой
        if len(d) == 13 and ean13_checksum_ok(d) and d not in runs13 \
                and not any(d in other for other in list(runs13) + list(near)):
            items.append(f"в текстовом слое PDF есть код {d} (контрольная цифра "
                         "сходится), но в прочитанных текстах его нет — "
                         "штрихкод не распознан или прочитан с ошибкой")
    for d, regs in near.items():
        where = ", ".join(dict.fromkeys(regs))
        items.append(f"число из {len(d)} цифр {d} (регионы: {where}) похоже "
                     "на неверно прочитанный штрихкод (EAN-13 — 13 цифр) — "
                     "требуется ручная проверка")
    if not items:
        if runs8:
            items = [f"возможен EAN-8: {d} (регионы: "
                     f"{', '.join(dict.fromkeys(regs))}) — может оказаться "
                     "датой или служебным числом, проверь глазами"
                     for d, regs in runs8.items()]
        else:
            items = ["штрихкод (13 цифр подряд) в прочитанных текстах не "
                     "найден — проверь макет глазами: возможно, цифры не "
                     "распознаны"]
    return {"id": 18, "key": "barcode", "name": "Штрихкод", "items": items}


SPELLING_PROMPT = """Ты — корректор русского текста упаковки пищевой продукции.
Найди орфографические и пунктуационные дефекты: ошибки в словах (в том числе
похожее по написанию, но не то слово — «молодой перец» вместо «молотый»,
«мороженные» вместо «мороженые»), пропущенные/лишние пробелы, незакрытые
скобки и кавычки, пропущенные точки, одно и то же слово или термин в РАЗНЫХ
написаниях на одном макете.
Для каждого дефекта заполни поля: wrong — дословно как на макете; correct —
как должно быть (единственный правильный вариант); what — суть дефекта
(3–10 слов). Для слова в двух написаниях в what перечисли обе формы
дословно, в correct — норму. Не используй слова «разнобой»,
«неконсистентно». Е-коды и алфавит их буквы не комментируй — это
проверяется отдельно. Проверяй ТОЛЬКО русский текст: английские, корейские,
китайские слова и строки не комментируй вообще. НЕ дефекты: отсутствие
точки в конце заголовка, строки таблицы, подписи поля («Изготовлено и
упаковано:») или отдельной надписи; переносы строк внутри регионов и слова,
обрезанные краем блока (артефакты чтения макета). НЕ комментируй стиль,
содержание и соответствие регламентам. Верни JSON:
{"remarks": [{"region": "<id региона>",
              "quote": "<дословный фрагмент с дефектом>",
              "wrong": "<как на макете>", "correct": "<как должно быть>",
              "what": "<суть дефекта, 3-10 слов>"}]}
Если дефектов нет — {"remarks": []}."""


def layer_word_findings(layout: dict, cfg: dict | None = None) -> list[str]:
    """Детерминированные находки из текстового СЛОЯ PDF — без LLM.

    unread_layer_words — слова слоя, не найденные в vision-тексте. Урок
    сверки Дня 6: vision молча нормализует дефекты («Hалейте» с латинской H
    прочитан как «Налейте», опечатка «plaease» — как «please»), и дефект
    терялся. Группировка (REVIEW-LOG R-13/R-17): раньше каждое непрочитанное
    слово шло отдельной строкой — 49 «неточностей» на гёдза были одним
    непрочитанным разделом «Способ приготовления», замаскированным под
    орфографию. Теперь:
    (а) подмены слов при чтении (R-15, из регионов) — каждая отдельно, первыми:
        слово слоя похоже на прочитанное, но не совпадает — на макете,
        скорее всего, опечатка, которую vision «починил»;
    (б) гомоглифы (кир/лат-микс) — каждый отдельно: это дефекты макета;
    (в) остальные непрочитанные слова — ОДНОЙ строкой с примерами: это
        пропуск чтения, а не N дефектов; при низком покрытии слоя — с
        отсылкой к шагу 1."""
    import unicodedata
    findings = []
    paired = set()
    for r in layout.get("regions", []) or []:
        for pair in r.get("word_substitutions") or []:
            paired.add(pair["layer"].lower())
            if pair["kind"] == "homoglyph":
                findings.append(
                    f"гомоглифная опечатка в макете: слово «{pair['layer']}» смешивает "
                    f"кириллицу и латиницу (прочитано как «{pair['vision']}» — дефект "
                    f"виден только в текстовом слое PDF; регион {r['id']})")
            else:
                findings.append(
                    f"возможная опечатка на макете: в текстовом слое PDF «{pair['layer']}», "
                    f"прочитано «{pair['vision']}» — vision мог молча исправить ошибку "
                    f"макета, сверьте с макетом (регион {r['id']})")
    rest = []
    for word in layout.get("unread_layer_words", []) or []:
        if not any(ch.isalpha() for ch in word) or len(word) < 4:
            continue  # числа и короткий шум (коды Pantone и т.п.)
        if word.lower() in paired:
            continue  # уже показано как подмена
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
            rest.append(word)
    if rest:
        cov = layout.get("text_layer_coverage")
        threshold = (cfg or {}).get("vision", {}).get("coverage_warning_threshold", 0.9)
        examples = ", ".join(rest[:8]) + (" …" if len(rest) > 8 else "")
        if isinstance(cov, (int, float)) and cov < threshold:
            findings.append(
                f"{len(rest)} слов текстового слоя PDF не найдены в прочитанном тексте "
                f"(покрытие слоя {cov:.0%}; например: {examples}) — вероятно, блок "
                "не прочитан или обрезан: это пропуск распознавания, а не орфография; "
                "проверьте и дочитайте блок на шаге 1")
        else:
            findings.append(
                f"{len(rest)} слов текстового слоя PDF не найдены в прочитанном тексте "
                f"(например: {examples}) — возможные опечатки макета или пропуск "
                "чтения; полный список — на шаге 1 в сверке блока")
    return findings


_LATIN_ECODE = re.compile(r"(?<![A-Za-z])E(\d{3,4}[a-zA-Z]?)(?![A-Za-z0-9])")


def ecode_alphabet_findings(facts: list[dict]) -> list[str]:
    """Е-коды латинской «E» в русских блоках — кодом, без LLM (REVIEW-LOG
    R-20: модель писала «разнобой в записи Е-кодов», не называя ни кода, ни
    нормы). В русской маркировке индекс добавки пишется кириллической «Е»;
    английские блоки не проверяются — там латиница уместна. Одна строка на
    регион, коды перечислены с правкой: E1414 → Е1414."""
    findings = []
    for f in facts:
        if f.get("lang") not in ("ru", "mixed"):
            continue
        codes = []
        for m in _LATIN_ECODE.finditer(f.get("text") or ""):
            code = "E" + m.group(1)
            if code not in codes:
                codes.append(code)
        if codes:
            fixes = ", ".join(f"{c} → Е{c[1:]}" for c in codes)
            findings.append(f"Е-код записан латинской буквой E вместо кириллической Е: "
                            f"{fixes} (регион {f['region']})")
    return findings


def spelling_check(facts: list[dict], client, cfg: dict,
                   tally: TokenTally, layout: dict | None = None) -> dict:
    """Аспект 19: детерминированные находки из слоя + CHEAP-модель по
    русским блокам макета.

    Голосование (REVIEW-LOG R-25): один вызов дешёвой модели давал на одном
    и том же макете разные наборы находок от прогона к прогону (то точки,
    то пробелы). Теперь `spelling_votes` вызовов, в отчёт идут только
    находки, повторившиеся не меньше чем в половине ответов (ключ — регион +
    дефектный фрагмент). Формат пункта (R-28): «что не так: «как на макете»
    → правильно: «как должно быть»» — читателю не нужно угадывать, какой из
    двух вариантов верный."""
    layer_items = layer_word_findings(layout or {}, cfg) + ecode_alphabet_findings(facts)
    ru_facts = [f for f in facts if f["lang"] in ("ru", "mixed")]
    base = {"id": 19, "key": "spelling", "name": "Орфография и пунктуация RU"}
    if not ru_facts:
        return {**base, "items": layer_items or ["русских блоков в макете не найдено"]}
    model = os.environ[cfg["verdict"]["cheap_env"]]
    votes = max(1, int(cfg["verdict"].get("spelling_votes", 3)))
    runs = []
    for _ in range(votes):
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
        runs.append([r for r in remarks if isinstance(r, dict)])
    items = [format_spelling_remark(r)
             for r in filter_spelling_remarks(consensus_remarks(runs))]
    items = layer_items + items
    return {**base, "items": items or ["дефектов не найдено"]}


def _remark_key(r: dict) -> tuple:
    frag = norm_text(str(r.get("wrong") or r.get("quote") or r.get("what") or ""))
    return (str(r.get("region", "?")), frag)


def consensus_remarks(runs: list[list[dict]]) -> list[dict]:
    """Находки, встретившиеся не меньше чем в половине прогонов (при одном
    прогоне — все). Порядок — как в первом прогоне, где находка появилась."""
    need = (len(runs) + 1) // 2
    seen: dict[tuple, dict] = {}
    counts: dict[tuple, int] = {}
    order: list[tuple] = []
    for run in runs:
        keys_in_run = set()
        for r in run:
            k = _remark_key(r)
            if k in keys_in_run:
                continue
            keys_in_run.add(k)
            counts[k] = counts.get(k, 0) + 1
            if k not in seen:
                seen[k] = r
                order.append(k)
    return [seen[k] for k in order if counts[k] >= need]


_CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)
_TRAILING_PUNCT = ".:;,!…"


def filter_spelling_remarks(remarks: list[dict]) -> list[dict]:
    """Страховка кодом поверх промпта (REVIEW-LOG R-25, проверка 04.09):
    модель, несмотря на запрет, стабильно (в 2 из 3 голосов) комментировала
    английские и корейские строки («Name :», «cabb → cabbage», «조리예») и
    «пропущенную точку» в заголовках и подписях полей («Изготовлено и
    упаковано:» → «…упаковано.»). Оставляем только находки, где дефектный
    фрагмент содержит кириллицу, и отбрасываем те, где правка сводится к
    знаку в конце строки."""
    out = []
    for r in remarks:
        frag = str(r.get("wrong") or r.get("quote") or "")
        if not _CYRILLIC.search(frag):
            continue
        wrong = str(r.get("wrong") or "").strip()
        correct = str(r.get("correct") or "").strip()
        if wrong and correct and wrong.rstrip(_TRAILING_PUNCT) == correct.rstrip(_TRAILING_PUNCT):
            continue
        out.append(r)
    return out


def format_spelling_remark(r: dict) -> str:
    """«что не так: «как на макете» → правильно: «как должно быть» — «цитата»
    (регион)». Старый формат ответа (без wrong/correct) — как раньше."""
    what = str(r.get("what", "?")).strip().rstrip(".")
    region = r.get("region", "?")
    quote = f" — «{r['quote']}»" if r.get("quote") else ""
    wrong, correct = (r.get("wrong") or "").strip(), (r.get("correct") or "").strip()
    if wrong and correct:
        return f"{what}: «{wrong}» → правильно: «{correct}»{quote} (регион {region})"
    if correct:
        return f"{what} → правильно: «{correct}»{quote} (регион {region})"
    return f"{what}{quote} (регион {region})"


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
                 categories_override: set[str] | None = None,
                 use_cache: bool = False, progress_cb=None) -> dict:
    """Все вердикты по макету → структура отчёта (сериализуемый dict).

    categories_override — решение Сергея («кнопки»): None = автодетект по
    маркерам; set() = только горизонтальные регламенты; {"poultry", ...} =
    заданные категории (в отчёте помечается source=manual).
    search_fn(query, regs) -> [(chunk_id, score)] — подменяется в тестах;
    по умолчанию — гибрид с query rewriting.
    use_cache — кэш ответов модели (data/verdict_cache.json): в библиотеке
    по умолчанию ВЫКЛЮЧЕН (тесты и замеры стабильности детерминированы без
    оглядки на кэш); CLI check включает его по умолчанию, флаг --no-cache
    выключает.
    progress_cb(done, total, name) — необязательный колбэк прогресса
    (UI показывает, какой аспект проверяется; done — сколько уже готово).
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
    total = len(aspects_data["aspects"])
    for i, aspect in enumerate(aspects_data["aspects"]):
        if progress_cb:
            progress_cb(i, total, f"{aspect['id']}. {aspect['name']}")
        if aspect["group"] == "regulatory":
            verdicts.append(judge_aspect(aspect, facts, categories, chunks,
                                         search_fn, client, cfg, tally,
                                         scope_basis, use_cache=use_cache))
        elif aspect["key"] == "barcode":
            other.append(barcode_check(layout))
        elif aspect["key"] == "spelling":
            other.append(spelling_check(facts, client, cfg, tally, layout))
    if progress_cb:
        progress_cb(total, total, "готово")

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
        if v.get("votes"):
            lines += ["", f"*Голосование ({len(v['votes'])} вызова): "
                          + ", ".join(v["votes"]) + ".*"]
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
