# -*- coding: utf-8 -*-
"""Vision-чтение PDF-макета упаковки: два прохода + сверка с текстовым слоем.

Схема (решения Дня 4, обоснование и замеры — docs/PIPELINE.md):

  PDF ─ render.py ─► полноразмерный PNG (scale 4) + обзорная копия (1568 px)
     │
  [1] ОБЗОР (MAIN_MODEL, 1 вызов): карта регионов — kind, язык, рамка 0..999.
     │   Полная модель: дешёвая на развёртках промахивается рамками до 15%.
     │
  [2] ЧТЕНИЕ (VISION_MODEL, вызов на регион): кроп полного разрешения с
     │   паддингом; дословная транскрипция; «[неразборчиво]» вместо догадок.
     │   technical-регионы (обвязка дизайнера) не читаются.
     │
  [3] СВЕРКА с текстовым слоем PDF (без LLM): слова слоя в зоне региона,
     │   не найденные в vision-тексте, выше порога → «требует ручной проверки».
     │   Слой в промпт НЕ подмешивается (может содержать невидимый/устаревший
     │   текст под кривыми — якорить модель на него опасно).
     │
  [4] JSON макета: regions[] + missing[] (обязательные kind без региона)
         + meta (модели, токены, SHA256 исходного PDF).

Статусы региона: «прочитано» / «требует ручной проверки».
Правило проекта: додуманный текст хуже отсутствующего.

CLI:  python -m labelcheck.vision <макет.pdf> [-o выход.json]
"""

import argparse
import base64
import hashlib
import io
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from labelcheck import render

STATUS_OK = "прочитано"
STATUS_MANUAL = "требует ручной проверки"
UNREADABLE_MARK = "[неразборчиво]"

# ── промпты ──────────────────────────────────────────────────────────────────

OVERVIEW_PROMPT = """Ты анализируешь макет упаковки пищевой продукции \
(дизайнерский PDF, отрендеренный в изображение; может быть развёрткой с \
несколькими панелями).

Найди ВСЕ содержательные блоки и верни JSON без пояснений:
{"regions": [{"id": "r1", "kind": "<категория>", "lang": "<ru|en|kk|other|mixed|none>",
              "bbox": [x0, y0, x1, y1], "note": "<1-6 слов, что это>"}]}

bbox — ЦЕЛЫЕ числа 0..999: доли ширины/высоты изображения, умноженные на 1000.
x0,y0 — левый верхний угол, x1,y1 — правый нижний.

Категории kind:
%s
- technical — техническая обвязка дизайнера: метки реза, цветовые плашки,
  размерные схемы, служебные таблицы макета, пустые шаблоны. Эти блоки
  ТОЖЕ верни — мы их отсечём.

Правила: не пропускай мелкий текст; блоки одного типа на разных языках —
отдельные регионы; не выдумывай блоки, которых не видишь."""

KIND_HINTS = {
    "product_name": "название продукта",
    "composition": "состав / ингредиенты",
    "nutrition": "пищевая и энергетическая ценность",
    "net_weight": "масса нетто / объём",
    "dates_storage": "даты, срок годности, условия хранения",
    "manufacturer": "изготовитель / импортёр / адреса",
    "marks": "знаки: ЕАС, штрихкод, пиктограммы переработки и т.п.",
    "usage": "способ приготовления / рекомендации",
    "other_text": "прочий содержательный текст",
}

READ_PROMPT = """На изображении — фрагмент макета упаковки пищевой продукции.
Перепиши ВЕСЬ текст фрагмента дословно, построчно, ничего не пропуская и не переводя.
Сохраняй Е-коды, проценты, номера, адреса и регистр как напечатано.
Текст, идущий по дуге, кругу, вертикально или под углом, тоже переписывай.
Знаки и пиктограммы, которые РЕАЛЬНО видны в кадре, опиши одной строкой каждую
в виде «знак: <что изображено или какая надпись внутри>»; штрихкод — строкой
«штрихкод: <цифры>». Не называй знаки по памяти или по образцу: если знака
в кадре нет — строки о нём быть не должно. Цифры под штрихкодом переписывай
только если они читаются целиком; иначе пиши «штрихкод: %s» — не додумывай
цифры.
Слова и строки, обрезанные КРАЕМ фрагмента, пропускай молча — не переписывай
и не помечай: это край выреза, а не дефект макета.
Пометку %s ставь только там, где текст целиком в кадре, но прочитать его нельзя.
НЕ угадывай нечитаемое. Никаких комментариев — только содержимое фрагмента.""" % (UNREADABLE_MARK, UNREADABLE_MARK)


# ── вспомогательные ──────────────────────────────────────────────────────────

def _b64(img):
    """PIL-изображение → base64-строка PNG для передачи в API."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _image_message(prompt, img, detail):
    """Собирает мультимодальное сообщение: текст-инструкция + изображение."""
    return [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{_b64(img)}",
                       "detail": detail}},
    ]}]


def sha256_file(path):
    """SHA256 файла — пломба исходного PDF в паспорте результата."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class TokenTally:
    """Счётчик токенов по моделям за прогон (основа счётчика стоимости, День 9)."""

    def __init__(self):
        self.by_model = {}

    def add(self, model, usage):
        d = self.by_model.setdefault(model, {"prompt": 0, "completion": 0, "calls": 0})
        d["prompt"] += usage.prompt_tokens
        d["completion"] += usage.completion_tokens
        d["calls"] += 1


# ── проход 1: обзор ──────────────────────────────────────────────────────────

def run_overview(client, model, full_img, cfg, tally):
    """Обзорный проход: карта регионов макета.

    Обычный макет — один вызов; сильно вытянутая лента (аспект больше
    overview_split_aspect) — обзор по половинам с перехлёстом: на обзоре
    целой ленты мелкие блоки пропадают (пропущенный блок изготовителя
    тестовая лента — замер Дня 4). Регионы половинок сводятся в координаты страницы,
    дубли из зоны перехлёста снимаются по перекрытию (IoU).

    Возвращает список регионов с нормализованными рамками 0..1.
    Регион с битой рамкой не выбрасывается молча — помечается manual.
    """
    kinds_text = "\n".join(f"- {k} — {KIND_HINTS[k]}" for k in cfg["kinds"]
                           if k != "technical")
    prompt = OVERVIEW_PROMPT % kinds_text
    if "overview_max_downscale" in cfg:
        tiles = render.overview_grid(
            full_img, cfg["overview_long_side"], cfg["overview_max_downscale"],
            cfg["crop_overlap_pct"], cfg.get("overview_tile_tolerance", 0.15))
    else:  # старое правило «по вытянутости» — для конфигов без нового ключа
        tiles = render.overview_tiles(full_img, cfg["overview_split_aspect"],
                                      cfg["crop_overlap_pct"])
    regions = []
    for t_i, (tile_img, frame) in enumerate(tiles):
        overview_img = render.make_overview(tile_img, cfg["overview_long_side"])
        resp = client.chat.completions.create(
            model=model,
            messages=_image_message(prompt, overview_img, detail="high"),
            response_format={"type": "json_object"},
        )
        tally.add(model, resp.usage)
        data = json.loads(resp.choices[0].message.content)
        for i, r in enumerate(data.get("regions", [])):
            region = {
                "id": f"t{t_i + 1}{r.get('id') or 'r%d' % (i + 1)}"
                      if len(tiles) > 1 else (r.get("id") or f"r{i + 1}"),
                "kind": r.get("kind", "other_text"),
                "lang": r.get("lang", "none"),
                "note": r.get("note", ""),
            }
            try:
                local = render.normalize_bbox(r["bbox"])
                region["bbox"] = render.tile_bbox_to_page(local, frame)
                region["bbox_ok"] = True
            except (KeyError, TypeError, ValueError):
                region["bbox"] = None
                region["bbox_ok"] = False
            regions.append(region)
    return dedup_regions(regions, cfg["dedup_iou"])


def dedup_regions(regions, iou_threshold):
    """Снимает дубли регионов из зоны перехлёста половинок обзора.

    Два региона одного kind с перекрытием рамок выше порога — один блок,
    увиденный дважды; остаётся первый.
    """
    kept = []
    for r in regions:
        if r["bbox_ok"] and any(
                k["bbox_ok"] and k["kind"] == r["kind"]
                and render.bbox_iou(k["bbox"], r["bbox"]) >= iou_threshold
                for k in kept):
            continue
        kept.append(r)
    return kept


# ── проход 2: чтение регионов ────────────────────────────────────────────────

def read_region(client, model, full_img, region, cfg, tally, pad_override=None):
    """Читает один регион кропами полного разрешения.

    Кропов обычно один; больше — если регион пришлось резать (max_crop_px).
    pad_override — паддинг вместо конфигного (для повторной попытки).
    Возвращает (текст, статус, список пиксельных рамок кропов).
    """
    crop_cfg = cfg if pad_override is None else {**cfg, "pad_pct": pad_override}
    pieces = render.crop_region(full_img, region["bbox"], crop_cfg)
    if not pieces:
        return "", STATUS_MANUAL, []
    texts, boxes = [], []
    for img, box_px in pieces:
        resp = client.chat.completions.create(
            model=model,
            messages=_image_message(READ_PROMPT, img, detail="high"),
        )
        tally.add(model, resp.usage)
        texts.append((resp.choices[0].message.content or "").strip())
        boxes.append(box_px)
    text = "\n".join(texts)
    # Пустое чтение — тоже manual: обзор видел здесь блок, а читать нечего,
    # значит рамка могла сместиться в пустое место.
    if not text.strip():
        return text, STATUS_MANUAL, boxes
    status = STATUS_MANUAL if UNREADABLE_MARK.lower() in text.lower() else STATUS_OK
    return text, status, boxes


def _needs_retry(text):
    """Первая попытка чтения провалилась: пусто или сплошное [неразборчиво].

    Такое чаще всего означает «рамка обзора уехала в пустое место» (мелкие
    блоки — самые чувствительные к сдвигу), а не «текст нечитаем»: повтор
    с бОльшим паддингом дёшев и часто спасает регион.
    """
    stripped = text.strip()
    return (not stripped
            or stripped.lower().replace(UNREADABLE_MARK.lower(), "").strip() == "")


# ── проход 3: сверка с текстовым слоем ───────────────────────────────────────

def _words(text, min_len):
    """Текст → множество значимых слов (нижний регистр, кириллица/латиница/цифры).

    Е-коды приводятся к кириллице (e621 → е621), как в токенизаторе BM25:
    в слое и в vision-тексте алфавит Е-кода может различаться, это не
    расхождение по существу.
    """
    words = re.findall(r"[а-яёa-z0-9가-힣]+", text.lower())
    # 'cid' — мусор текстового слоя: pdfplumber пишет «(cid:NNN)» для
    # символов без юникод-маппинга, это не слова макета.
    # Хангыль плотный (당면 = «стеклянная лапша» — 2 слога), поэтому
    # корейские слова проходят с длины 2 — иначе сверка слепа к KR
    # (потеря 대두단백/당면 в прогоне v3 прошла мимо сторожей).
    def keep(w):
        if w == "cid":
            return False
        if re.search(r"[가-힣]", w):
            return len(w) >= 2
        return len(w) >= min_len
    return {re.sub(r"^e(\d)", r"е\1", w) for w in words if keep(w)}


def text_layer_words(pdf_path, boxes_px, scale, min_len,
                     inset_px=(0, 0), page_index=0):
    """Слова текстового слоя PDF внутри пиксельных рамок кропов.

    Берутся только слова, ЦЕЛИКОМ лежащие в «ядре» рамки — рамке, сжатой
    на inset_px (паддинговая кайма кропа). Модель по промпту пропускает
    текст, обрезанный краем кропа, поэтому слова из каймы нельзя требовать
    с неё — каждый сосед в кайме давал ложное расхождение (тестовая лента, День 4).

    Ядро считается от ОБЪЕДИНЕНИЯ кропов, а не от каждого куска: длинный
    регион режется на половины с перехлёстом, и кайма каждой половины
    выбрасывала целую полосу слов на стыке (гёдза, REVIEW-LOG R-15:
    «молодой» лежал в зазоре между ядрами половин и в сверку не попадал —
    подмена «молодой → молотый» была невидима). Перехлёст половин
    гарантирует, что слово на стыке целиком есть хотя бы в одной из них.
    Возвращает None, если текстового слоя в зоне нет (макет в кривых).
    """
    import pdfplumber

    if not boxes_px:
        return None
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_index]
        # dedupe_chars: надписи с обводкой лежат в PDF дважды поверх себя,
        # без дедупликации extract_words склеивает буквы («ccllaassssiicc») —
        # каждое такое слово давало ложное расхождение сверки.
        words = page.dedupe_chars().extract_words()
        page.flush_cache()
        page.get_textmap.cache_clear()
    return layer_words_in_core(words, boxes_px, scale, min_len, inset_px)


def layer_words_in_core(words, boxes_px, scale, min_len, inset_px=(0, 0)):
    """Чистая часть text_layer_words: слова pdfplumber (x0/top/x1/bottom в
    пунктах) → множество слов, целиком лежащих в ядре объединения кропов."""
    ix, iy = inset_px
    cx0 = min(b[0] for b in boxes_px) + ix
    cy0 = min(b[1] for b in boxes_px) + iy
    cx1 = max(b[2] for b in boxes_px) - ix
    cy1 = max(b[3] for b in boxes_px) - iy
    found = set()
    for w in words:
        wx0, wy0, wx1, wy1 = (w["x0"] * scale, w["top"] * scale,
                              w["x1"] * scale, w["bottom"] * scale)
        if wx0 >= cx0 and wy0 >= cy0 and wx1 <= cx1 and wy1 <= cy1:
            found |= _words(w["text"], min_len)
    return found if found else None


def page_coverage(pdf_path, all_vision_text, cfg, page_index=0):
    """Глобальный coverage-чек: какая доля слов текстового слоя ВСЕЙ страницы
    прочитана vision хоть каким-нибудь регионом.

    Ловит блок, целиком пропущенный обзором (случай тестовой ленты, День 4): рамки и
    kind-метки могут врать, а непрочитанные слова страницы врать не могут.
    Урок coverage основного проекта: найти «что-то» — не значит найти «всё».
    Возвращает (coverage 0..1 | None, примеры непрочитанных слов).
    None — слоя нет, чек невозможен (это НЕ ошибка).
    """
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_index]
        layer_text = page.dedupe_chars().extract_text() or ""
        page.flush_cache()
        page.get_textmap.cache_clear()
    layer = _words(layer_text, cfg["min_word_len"])
    if not layer:
        return None, []
    seen = _words(all_vision_text, cfg["min_word_len"])
    joined = re.sub(r"[^а-яёa-z0-9]+", "", all_vision_text.lower())
    unread = sorted(w for w in layer - seen if w not in joined)
    return 1.0 - len(unread) / len(layer), unread


def check_against_text_layer(vision_text, layer_words, cfg):
    """Доля слов слоя, не найденных в vision-тексте.

    Слово слоя, не найденное среди слов vision-текста, дополнительно ищется
    ПОДСТРОКОЙ в слитом тексте: дизайнерский трекинг рвёт слова слоя на куски
    («chic» + «ken»), а vision пишет слово целиком — это не расхождение.

    Возвращает (mismatch_ratio | None, список расхождений).
    None — сверка невозможна (слоя нет), это НЕ ошибка.
    """
    if not layer_words:
        return None, []
    seen = _words(vision_text, cfg["min_word_len"])
    joined = re.sub(r"[^а-яёa-z0-9]+", "", vision_text.lower())
    missing = sorted(w for w in layer_words - seen if w not in joined)
    return len(missing) / len(layer_words), missing


def missing_kinds(result_regions, cfg):
    """Обязательные типы блоков, не найденные НИ меткой kind, НИ сигнальными
    словами в прочитанных текстах.

    Kind-метки обзора недетерминированы (тот же состав приходит то
    composition, то other_text) — содержимое надёжнее метки.
    """
    present = {r["kind"] for r in result_regions}
    all_text = " ".join(r["text"] for r in result_regions).lower()
    markers = cfg.get("kind_markers", {})
    out = []
    for kind in cfg["required_kinds"]:
        if kind in present:
            continue
        if any(m in all_text for m in markers.get(kind, [])):
            continue
        out.append(kind)
    return out


# ── сборка ───────────────────────────────────────────────────────────────────

def layer_word_boxes(pdf_path, page_index=0):
    """Слова текстового слоя с рамками в долях страницы 0..1.

    Для прищёлкивания рамок обзора и двустороннего сторожа. None — слоя нет
    (макет в кривых). dedupe_chars — как в text_layer_words (обводка дублирует
    буквы).
    """
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_index]
        pw, ph = float(page.width), float(page.height)
        words = page.dedupe_chars().extract_words()
        page.flush_cache()
        page.get_textmap.cache_clear()
    out = [(w["text"], (w["x0"] / pw, w["top"] / ph,
                        w["x1"] / pw, w["bottom"] / ph)) for w in words]
    return out or None


def snap_bbox_to_words(bbox, word_boxes, max_expand_pct):
    """Расширяет рамку региона до ЦЕЛЫХ слов текстового слоя.

    Рамки обзора гуляют и режут слова по краю (разметка Дня 8: «SEOUL» без
    хвоста, склейки на краях). Слово, чей центр внутри рамки, должно попасть
    в неё целиком. Рамка только расширяется (не сжимается — растровые зоны
    слоя не имеют) и не дальше max_expand_pct страницы за сторону, чтобы
    не утащить соседний блок.
    """
    x0, y0, x1, y1 = bbox
    m = max_expand_pct / 100.0
    nx0, ny0, nx1, ny1 = x0, y0, x1, y1
    for _, (wx0, wy0, wx1, wy1) in word_boxes:
        cx, cy = (wx0 + wx1) / 2, (wy0 + wy1) / 2
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            nx0, ny0 = min(nx0, wx0), min(ny0, wy0)
            nx1, ny1 = max(nx1, wx1), max(ny1, wy1)
    return (max(x0 - m, max(0.0, nx0)), max(y0 - m, max(0.0, ny0)),
            min(x1 + m, min(1.0, nx1)), min(y1 + m, min(1.0, ny1)))


def ink_ratio(full_img, bbox):
    """Доля «чернильных» пикселей в рамке (отклонение от фонового тона).

    Сторож пустых рамок (разметка Дня 8: пустой жёлтый регион mandu получил
    чужой текст со словом «Exporter» из повторного чтения расширенной рамки).
    """
    W, H = full_img.size
    x0, y0, x1, y1 = bbox
    box = (round(x0 * W), round(y0 * H), round(x1 * W), round(y1 * H))
    if box[2] - box[0] < 4 or box[3] - box[1] < 4:
        return 0.0
    crop = full_img.crop(box).convert("L")
    crop.thumbnail((256, 256))
    hist = crop.histogram()
    modal = max(range(256), key=lambda v: hist[v])
    total = sum(hist)
    ink = sum(n for v, n in enumerate(hist) if abs(v - modal) > 28)
    return ink / total if total else 0.0


_INVENT_WORD = re.compile(r"[а-яёa-z]{4,}|[가-힣]{2,}", re.IGNORECASE)
# Строки-описания графики (модель пишет их по промпту) — не текст макета,
# в слое их нет по определению; в сторож выдумок не идут.
_SIGN_LINE = re.compile(r"^\s*(знак|штрихкод|пиктограмм|логотип|иконк|\[)",
                        re.IGNORECASE)


def invented_words(vision_text, page_words, min_len=4):
    """Слова vision-текста, которых нет нигде в текстовом слое страницы.

    Двусторонний сторож (разметка Дня 8: сфабрикованный KR-состав, «Exporter»,
    «PENTHIOPIL»). Применять только к регионам, где слой есть: в растровых
    зонах отсутствие слова в слое ничего не значит. Не участвуют: числа и
    короткие слова (шум), строки-описания знаков («знак EAC» — их нет в слое
    по определению), слова, найденные ПОДСТРОКОЙ в склейке слоя — разреженные
    заголовки («П и щ е в а я») слой отдаёт побуквенно.
    """
    def norm(w):
        return w.lower().replace("ё", "е")
    known = set()
    for text, _ in page_words:
        for w in _INVENT_WORD.findall(text):
            known.add(norm(w))
    joined = norm("".join(t for t, _ in page_words))
    out = []
    for w in invent_candidates(vision_text, min_len):
        if norm(w) not in known and norm(w) not in joined:
            out.append(w)
    return out


def invent_candidates(vision_text, min_len=4):
    """Слова vision-текста, которые сторож выдумок вообще проверяет (без
    строк-описаний знаков, без коротких слов). Знаменатель для доли
    «слов вне слоя» по региону и по странице (REVIEW-LOG R-23)."""
    checked = "\n".join(line for line in (vision_text or "").splitlines()
                         if not _SIGN_LINE.match(line))
    return [w for w in _INVENT_WORD.findall(checked)
            if len(w) >= (2 if re.search(r"[가-힣]", w) else min_len)]


_DIGIT_RUN_SPACED = re.compile(r"\d(?:[\d ]*\d)?")


def digit_runs(text, min_len=8):
    """Длинные числа текста (пробелы внутри снимаются: «8 805957 025951» —
    один код). Штрихкоды, партии, телефоны — то, что сторож букв не видит
    (REVIEW-LOG R-16: выдуманный EAN в конце обрезанного блока)."""
    out = []
    for m in _DIGIT_RUN_SPACED.finditer(text or ""):
        d = m.group().replace(" ", "")
        if len(d) >= min_len:
            out.append(d)
    return out


def layer_digit_runs(page_words, min_len=8):
    """Длинные числа текстового слоя всей страницы — эталон для сверки
    штрихкода. Слова слоя идут в порядке извлечения, поэтому разбитый
    пробелами код склеивается. Пусто — цифр в слое нет (штрихкод картинкой),
    сверка невозможна, это не ошибка."""
    if not page_words:
        return []
    return digit_runs(" ".join(t for t, _ in page_words), min_len)


def invented_digits(vision_text, layer_runs, min_len=8):
    """Числа vision-текста, которых нет в числах слоя (только если в слое
    вообще есть длинные числа — иначе сравнивать не с чем)."""
    if not layer_runs:
        return []
    return [d for d in digit_runs(vision_text, min_len)
            if not any(d in run for run in layer_runs)]


# Латинские буквы, неотличимые на глаз от кириллических (гомоглифы):
# «Hалейте» с латинской H — дефект макета, который vision молча «чинит».
_HOMOGLYPHS = str.maketrans("abcehkmoptxyABCEHKMOPTXY", "авсенкмортхуАВСЕНКМОРТХУ")


def levenshtein(a, b, limit=3):
    """Редакционное расстояние с ранним выходом (> limit → limit + 1)."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > limit:
            return limit + 1
        prev = cur
    return prev[-1]


def word_substitutions(missing_words, invented, cfg):
    """Пары «слово слоя не прочитано ↔ слово vision не из слоя», похожие
    друг на друга: vision прочитал НЕ то, что напечатано (REVIEW-LOG R-15:
    «молодой» → «молотый» — модель молча исправила опечатку макета).
    kind: homoglyph — те же буквы, но часть латиницей (дефект макета);
    substitution — другое слово. Каждое слово vision участвует один раз,
    берётся ближайшая пара."""
    max_d = cfg.get("substitution_max_distance", 2)
    min_len = cfg.get("substitution_min_len", 5)
    free = [w for w in invented if len(w) >= min_len]
    pairs = []
    for m in missing_words:
        if len(m) < min_len:
            continue
        best, best_d = None, max_d + 1
        for v in free:
            d = levenshtein(m.lower(), v.lower(), max_d)
            if d < best_d:
                best, best_d = v, d
        if best is None or best_d > max_d:
            continue
        free.remove(best)
        same = m.lower().translate(_HOMOGLYPHS) == best.lower().translate(_HOMOGLYPHS)
        pairs.append({"layer": m, "vision": best,
                      "kind": "homoglyph" if same else "substitution"})
    return pairs


def merge_regions(regions, gap):
    """Склейка примыкающих рамок в крупные блоки (итог перепрогона v2).

    На чётких тайлах обзор начал дробить большой текстовый блок на
    «ленточки»-строки с дырами между ними (EN-состав mandu выпал в дыру).
    Рамки, пересекающиеся или отстоящие меньше gap (доля страницы),
    объединяются: крупный блок — проверенный режим чтения (кропы >2000px
    режутся на части штатно). technical и битые рамки не трогаются.
    """
    def close(a, b):
        return (a[0] - gap <= b[2] and b[0] - gap <= a[2]
                and a[1] - gap <= b[3] and b[1] - gap <= a[3])
    regs = [dict(r) for r in regions]
    changed = True
    while changed:
        changed = False
        out = []
        for r in regs:
            if not r.get("bbox_ok") or r.get("kind") == "technical":
                out.append(r)
                continue
            hit = next((o for o in out
                        if o.get("bbox_ok") and o.get("kind") != "technical"
                        and close(r["bbox"], o["bbox"])), None)
            if hit is None:
                out.append(r)
                continue
            hb, rb = hit["bbox"], r["bbox"]
            hit["bbox"] = (min(hb[0], rb[0]), min(hb[1], rb[1]),
                           max(hb[2], rb[2]), max(hb[3], rb[3]))
            hit["note"] = (hit.get("note") or "") + f" +{r['id']}"
            changed = True
        regs = out
    return regs


def layer_fill_regions(regions, page_words, gap=0.015, min_words=3):
    """Регионы из текстового слоя для зон, которые обзор не покрыл.

    Слой знает, где на странице текст, лучше любой модели: слова, чей центр
    не попал ни в одну рамку, кластеризуются по близости и становятся
    дополнительными регионами (kind=other_text — kind это только подсказка).
    Гарантия полноты там, где слой есть; страховка от «дыр» обзора.
    """
    boxes = [r["bbox"] for r in regions
             if r.get("bbox_ok") and r.get("kind") != "technical"]

    def covered(cx, cy):
        return any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in boxes)
    orphans = [wb for _, wb in page_words
               if not covered((wb[0] + wb[2]) / 2, (wb[1] + wb[3]) / 2)]
    clusters = []
    for wb in sorted(orphans, key=lambda w: (w[1], w[0])):
        cx, cy = (wb[0] + wb[2]) / 2, (wb[1] + wb[3]) / 2
        hit = next((c for c in clusters
                    if c["bbox"][0] - gap <= cx <= c["bbox"][2] + gap
                    and c["bbox"][1] - gap <= cy <= c["bbox"][3] + gap), None)
        if hit is None:
            clusters.append({"bbox": list(wb), "n": 1})
        else:
            b = hit["bbox"]
            hit["bbox"] = [min(b[0], wb[0]), min(b[1], wb[1]),
                           max(b[2], wb[2]), max(b[3], wb[3])]
            hit["n"] += 1
    out = []
    for i, c in enumerate(clusters, start=1):
        if c["n"] < min_words:
            continue  # одиночные препресс-метки — не регион
        b = c["bbox"]
        out.append({"id": f"L{i}", "kind": "other_text", "lang": "mixed",
                    "note": "добавлен по текстовому слою (обзор зону не покрыл)",
                    "bbox": (max(0.0, b[0] - 0.004), max(0.0, b[1] - 0.004),
                             min(1.0, b[2] + 0.004), min(1.0, b[3] + 0.004)),
                    "bbox_ok": True})
    return out


def assess_region(text, read_status, bbox, boxes_px, full_img, pdf_path,
                  page_words, cfg, layer_runs=None):
    """Проход 3 для одного региона: сторожа чернил и текстового слоя.

    Чистая функция от (текст, статус чтения, рамка, кропы, рендер, PDF):
    её вызывает и живой прогон (analyze), и пересчёт статусов по готовому
    layout'у без API (reguard_layout). Сторож ВЫДУМОК здесь только собирает
    материал (список слов, счётчики) — решение о статусе принимает
    finalize_layer_guards по всей странице сразу (REVIEW-LOG R-23).
    """
    status = read_status
    if not text.strip():
        reason = "регион пуст — рамка обзора могла сместиться"
    elif status == STATUS_MANUAL:
        reason = "нечитаемые места"
    else:
        reason = ""
    # Сторож чернил: непустой текст при пустой рамке = текст пришёл из
    # соседней зоны (повторное чтение расширяет рамку) — не факт макета.
    if text.strip() and ink_ratio(full_img, bbox) < cfg.get("min_ink_ratio", 0.004):
        status = STATUS_MANUAL
        reason = ("в рамке нет чернил — текст мог прийти из соседней "
                  "зоны при повторном чтении")
    pad = cfg["pad_pct"] / 100.0
    inset = (pad * full_img.size[0], pad * full_img.size[1])
    layer = text_layer_words(pdf_path, boxes_px, cfg["render_scale"],
                             cfg["min_word_len"], inset_px=inset)
    mismatch, missing_words = check_against_text_layer(text, layer, cfg)
    if mismatch is not None and mismatch > cfg["text_layer_mismatch_threshold"]:
        status = STATUS_MANUAL
        reason = (f"расхождение с текстовым слоем "
                  f"{mismatch:.0%}: {', '.join(missing_words[:5])}")
    fake, candidates, pairs = [], 0, []
    if page_words and layer is not None:
        fake = invented_words(text, page_words)
        candidates = len(invent_candidates(text))
        # Подмена слова — сигнал высшего приоритета: vision прочитал не то,
        # что напечатано, и от кривых это не зависит (пара требует и слова
        # слоя, и похожего слова vision).
        pairs = word_substitutions(missing_words, fake, cfg)
        if pairs:
            status = STATUS_MANUAL
            note = "ВОЗМОЖНАЯ ПОДМЕНА СЛОВА: " + "; ".join(
                f"на макете «{p['layer']}», прочитано «{p['vision']}»"
                + (" (латиница вместо кириллицы)" if p["kind"] == "homoglyph" else "")
                for p in pairs[:4])
            reason = (note + "; " + reason) if reason else note
    digits = invented_digits(text, layer_runs or [])
    return {"status": status, "status_reason": reason,
            "text_layer_mismatch": mismatch,
            "layer_missing_words": missing_words[:200],
            "invented_words": fake[:100],
            "invented_digits": digits,
            "word_substitutions": pairs,
            "invent_candidates": candidates,
            "layer_words_in_box": len(layer) if layer else 0,
            "vision_words": len(_words(text, cfg["min_word_len"])),
            "has_layer": layer is not None}


def page_invented_share(regions):
    """Доля проверяемых vision-слов страницы, которых нет в текстовом слое
    (по регионам, где слой есть). None — таких регионов нет."""
    cand = sum(r.get("invent_candidates", 0) for r in regions if r.get("has_layer"))
    if not cand:
        return None
    fake = sum(len(r.get("invented_words") or []) for r in regions if r.get("has_layer"))
    return round(fake / cand, 3)


def page_layer_partial(regions, cfg):
    """Страница — «смесь кривых и живого текста»: слов вне слоя больше
    порога. Пересчёт по layout'ам приёмки 02.09: креветки 0.41 (шесть ложных
    «выдумок»), гёдза 0.11, манду 0.03 (там сторож ловил настоящие фабрикации).
    """
    share = page_invented_share(regions)
    return share is not None and share >= cfg.get("layer_partial_page_share", 0.2)


def region_layer_partial(region, cfg):
    """Регион частично в кривых: слов слоя в рамке заметно меньше, чем слов
    vision (логотип SEOUL на манду — 4 слова слоя на 15 прочитанных; блок
    изготовителя креветок — 3 на 25). В таком регионе отсутствие слова в слое
    ничего не значит."""
    if not region.get("has_layer"):
        return False
    ratio = cfg.get("layer_partial_ratio", 0.5)
    return region.get("layer_words_in_box", 0) < ratio * region.get("vision_words", 0)


def finalize_layer_guards(regions, cfg, skip_ids=()):
    """Сторож выдумок с учётом кривых (R-23). Регион уходит в ручную
    проверку по словам вне слоя ТОЛЬКО если ни страница, ни регион не
    частичны; иначе слова остаются в invented_words для таблицы
    расхождений, а статус не трогается. skip_ids — регионы, статус которых
    выставил человек (правка/подтверждение) — не пересматриваются."""
    page_partial = page_layer_partial(regions, cfg)
    for r in regions:
        if not r.get("has_layer"):
            r["layer_partial"] = False
            continue
        partial = page_partial or region_layer_partial(r, cfg)
        r["layer_partial"] = partial
        if r["id"] in skip_ids or partial:
            continue
        fake = r.get("invented_words") or []
        share = len(fake) / max(1, r.get("invent_candidates", 0))
        if (len(fake) >= cfg.get("invented_min_words", 2)
                and share >= cfg.get("invented_share", 0.08)):
            r["status"] = STATUS_MANUAL
            note = ("слова вне текстового слоя (возможная выдумка): "
                    + ", ".join(fake[:5]))
            r["status_reason"] = (r["status_reason"] + "; " + note
                                  if r.get("status_reason") else note)
    return regions


def reguard_layout(layout, pdf_path, cfg=None):
    """Пересчёт сторожей по готовому layout'у БЕЗ API: тексты остаются, статусы
    и поля сверки пересчитываются по PDF и рендеру. Регионы, которые человек
    правил или подтверждал, не трогаются. Нужен, чтобы применить новые
    правила сторожей (R-23) к уже распознанным макетам без перечитывания
    ($0.10 и новый недетерминизм зрения на каждый макет)."""
    cfg = cfg or render.load_config()
    full_img = render.render_page(pdf_path, cfg["render_scale"])
    page_words = layer_word_boxes(pdf_path)
    layer_runs = layer_digit_runs(page_words)
    human = {r["id"] for r in layout["regions"]
             if r.get("human_edited") or r.get("human_confirmed")}
    for r in layout["regions"]:
        text = r.get("text") or ""
        if not text.strip() and r.get("status_reason") == "битая рамка обзора":
            continue
        pieces = render.crop_region(full_img, r["bbox"], cfg)
        boxes_px = [box for _, box in pieces]
        read_status = (STATUS_MANUAL if UNREADABLE_MARK.lower() in text.lower()
                       or not text.strip() else STATUS_OK)
        fields = assess_region(text, read_status, r["bbox"], boxes_px,
                               full_img, pdf_path, page_words, cfg, layer_runs)
        if r["id"] in human:
            # статус человека — истина; счётчики сверки всё равно нужны,
            # иначе доля «слов вне слоя» по странице считалась бы без
            # самого большого (правленого) блока и искажалась
            fields.pop("status"), fields.pop("status_reason")
        r.update(fields)
    finalize_layer_guards(layout["regions"], cfg, skip_ids=human)
    layout["text_layer_partial"] = page_layer_partial(layout["regions"], cfg)
    layout["text_layer_invented_share"] = page_invented_share(layout["regions"])
    layout["layer_digit_runs"] = layer_runs
    all_text = "\n".join(r.get("text") or "" for r in layout["regions"])
    coverage, unread = page_coverage(pdf_path, all_text, cfg)
    layout["text_layer_coverage"] = coverage
    layout["unread_layer_words"] = unread[:300]
    return layout


def analyze(pdf_path, cfg=None, out_path=None, quiet=False):
    """Полный vision-прогон макета: PDF → JSON макета."""
    load_dotenv()
    cfg = cfg or render.load_config()
    overview_model = os.environ["MAIN_MODEL"]
    read_model = os.environ["VISION_MODEL"]
    client = OpenAI()
    tally = TokenTally()
    t0 = time.time()

    full_img = render.render_page(pdf_path, cfg["render_scale"])
    scale = cfg["render_scale"]

    regions = run_overview(client, overview_model, full_img, cfg, tally)
    if not quiet:
        print(f"обзор: {len(regions)} регионов")

    # Прищёлкивание рамок к словам текстового слоя: рамки обзора гуляют и
    # режут слова по краю; слово с центром внутри рамки входит в неё целиком.
    page_words = layer_word_boxes(pdf_path)
    if page_words:
        for region in regions:
            if region.get("bbox_ok"):
                region["bbox"] = snap_bbox_to_words(
                    region["bbox"], page_words,
                    cfg.get("snap_max_expand_pct", 5.0))
    regions = merge_regions(regions, cfg.get("merge_gap_pct", 0.8) / 100.0)
    if page_words:
        extra = layer_fill_regions(regions, page_words)
        if extra and not quiet:
            print(f"добавлено по слою: {len(extra)} регионов")
        regions += extra

    layer_runs = layer_digit_runs(page_words)
    result_regions = []
    for region in regions:
        if region["kind"] == "technical":
            continue
        if not region["bbox_ok"]:
            result_regions.append({**_public(region), "text": "",
                                   "status": STATUS_MANUAL,
                                   "status_reason": "битая рамка обзора"})
            continue
        text, status, boxes_px = read_region(
            client, read_model, full_img, region, cfg, tally)
        if _needs_retry(text):
            # рамка, скорее всего, уехала: расширяем зону от центра
            # до минимум retry_min_side_pct стороны + увеличенный паддинг
            wide = {**region, "bbox": render.expand_to_min(
                region["bbox"], cfg["retry_min_side_pct"])}
            text, status, boxes_px = read_region(
                client, read_model, full_img, wide, cfg, tally,
                pad_override=cfg["retry_pad_pct"])
        result_regions.append({**_public(region), "text": text,
                               **assess_region(text, status, region["bbox"],
                                               boxes_px, full_img, pdf_path,
                                               page_words, cfg, layer_runs)})
    finalize_layer_guards(result_regions, cfg)
    if not quiet:
        for r in result_regions:
            flag = "⚠" if r["status"] == STATUS_MANUAL else "✓"
            print(f"  {flag} {r['id']:4s} {r['kind']:14s} {r['lang']:5s} "
                  f"{len(r['text'])} симв.")

    missing = missing_kinds(result_regions, cfg)

    all_text = "\n".join(r["text"] for r in result_regions)
    coverage, unread = page_coverage(pdf_path, all_text, cfg)
    if not quiet and coverage is not None:
        print(f"покрытие текстового слоя: {coverage:.0%}"
              + (f", непрочитано напр.: {', '.join(unread[:8])}" if unread else ""))

    result = {
        "meta": {
            "source_pdf": os.path.basename(str(pdf_path)),
            "source_sha256": sha256_file(pdf_path),
            "render_scale": scale,
            "overview_model": overview_model,
            "read_model": read_model,
            "tokens": tally.by_model,
            "seconds": round(time.time() - t0, 1),
        },
        "regions": result_regions,
        "missing": missing,
        "text_layer_coverage": coverage,
        "unread_layer_words": unread[:300],
        "text_layer_partial": page_layer_partial(result_regions, cfg),
        "text_layer_invented_share": page_invented_share(result_regions),
        "layer_digit_runs": layer_runs,
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
    return result


def _public(region):
    """Поля региона, попадающие в итоговый JSON (без служебного bbox_ok)."""
    return {"id": region["id"], "kind": region["kind"], "lang": region["lang"],
            "note": region["note"], "bbox": region["bbox"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Vision-чтение PDF-макета упаковки")
    ap.add_argument("pdf", nargs="?", help="путь к PDF-макету")
    ap.add_argument("-o", "--out", default=None,
                    help="куда сохранить JSON (по умолчанию data/layouts/<имя>.json)")
    ap.add_argument("--reguard", metavar="LAYOUT_JSON", default=None,
                    help="не читать заново: пересчитать сторожа слоя по готовому "
                         "layout'у (без API), результат — в тот же файл")
    ap.add_argument("--reguard-all", action="store_true",
                    help="пересчитать сторожа по ВСЕМ layout'ам data/layouts/ "
                         "(кроме *.orig.json); PDF ищется по meta.source_pdf в "
                         "data/samples_private и data/ui_uploads")
    args = ap.parse_args(argv)
    if args.reguard_all:
        return _reguard_all()
    if not args.pdf:
        ap.error("укажите PDF-макет (или --reguard-all)")
    if args.reguard:
        from labelcheck.store import save_layout
        layout = json.loads(Path(args.reguard).read_text(encoding="utf-8"))
        before = {r["id"]: r["status"] for r in layout["regions"]}
        reguard_layout(layout, args.pdf)
        save_layout(layout, args.reguard)
        changed = [(rid, before[rid], r["status"]) for rid, r in
                   ((r["id"], r) for r in layout["regions"]) if before[rid] != r["status"]]
        manual = [r["id"] for r in layout["regions"] if r["status"] == STATUS_MANUAL]
        print(f"страница частично в кривых: {layout['text_layer_partial']} "
              f"(слов вне слоя: {layout['text_layer_invented_share']})")
        print(f"ручная проверка: {len(manual)} {manual or ''}; изменили статус: "
              + (", ".join(f"{rid} {a[:9]}→{b[:9]}" for rid, a, b in changed) or "—"))
        print(f"JSON: {args.reguard}")
        return 0
    out = args.out or Path("data/layouts") / (Path(args.pdf).stem + ".json")
    result = analyze(args.pdf, out_path=out)
    manual = [r["id"] for r in result["regions"] if r["status"] == STATUS_MANUAL]
    print(f"\nрегионов: {len(result['regions'])}, "
          f"ручная проверка: {len(manual)} {manual or ''}")
    if result["missing"]:
        print(f"⚠ не найдены обязательные блоки: {', '.join(result['missing'])}")
    t = result["meta"]["tokens"]
    for m, d in t.items():
        print(f"токены {m}: {d['prompt']}+{d['completion']} ({d['calls']} вызовов)")
    print(f"JSON: {out}")
    return 0


def _reguard_all(layouts_dir="data/layouts",
                 pdf_dirs=("data/samples_private", "data/ui_uploads")):
    """Пересчёт сторожей по всем layout'ам, у которых нашёлся исходный PDF
    (R-40: у одного PDF бывает два layout'а — канонический и копия по имени
    PDF; пересчёт по имени файла второй пропускал)."""
    from labelcheck.store import save_layout
    done = skipped = 0
    for path in sorted(Path(layouts_dir).glob("*.json")):
        if path.name.endswith(".orig.json"):
            continue
        layout = json.loads(path.read_text(encoding="utf-8"))
        name = (layout.get("meta") or {}).get("source_pdf")
        pdf = next((Path(d) / name for d in pdf_dirs if name and (Path(d) / name).exists()), None)
        if pdf is None:
            print(f"— {path.name}: PDF «{name}» не найден, пропуск")
            skipped += 1
            continue
        before = {r["id"]: r["status"] for r in layout["regions"]}
        reguard_layout(layout, str(pdf))
        save_layout(layout, path)
        changed = sum(1 for r in layout["regions"] if before[r["id"]] != r["status"])
        manual = sum(1 for r in layout["regions"] if r["status"] == STATUS_MANUAL)
        print(f"✓ {path.name}: ручная проверка {manual}, статус изменили {changed}, "
              f"страница частично в кривых: {layout['text_layer_partial']}")
        done += 1
    print(f"пересчитано {done}, пропущено {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
