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
Опиши обнаруженные знаки/пиктограммы одной строкой каждую (например: «знак EAC»,
«штрихкод: <цифры>»).
Слова и строки, обрезанные КРАЕМ фрагмента, пропускай молча — не переписывай
и не помечай: это край выреза, а не дефект макета.
Пометку %s ставь только там, где текст целиком в кадре, но прочитать его нельзя.
НЕ угадывай нечитаемое. Никаких комментариев — только содержимое фрагмента.""" % UNREADABLE_MARK


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
    words = re.findall(r"[а-яёa-z0-9]+", text.lower())
    # 'cid' — мусор текстового слоя: pdfplumber пишет «(cid:NNN)» для
    # символов без юникод-маппинга, это не слова макета
    return {re.sub(r"^e(\d)", r"е\1", w) for w in words
            if len(w) >= min_len and w != "cid"}


def text_layer_words(pdf_path, boxes_px, scale, min_len,
                     inset_px=(0, 0), page_index=0):
    """Слова текстового слоя PDF внутри пиксельных рамок кропов.

    Берутся только слова, ЦЕЛИКОМ лежащие в «ядре» рамки — рамке, сжатой
    на inset_px (паддинговая кайма кропа). Модель по промпту пропускает
    текст, обрезанный краем кропа, поэтому слова из каймы нельзя требовать
    с неё — каждый сосед в кайме давал ложное расхождение (тестовая лента, День 4).
    Возвращает None, если текстового слоя в зоне нет (макет в кривых).
    """
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_index]
        # dedupe_chars: надписи с обводкой лежат в PDF дважды поверх себя,
        # без дедупликации extract_words склеивает буквы («ccllaassssiicc») —
        # каждое такое слово давало ложное расхождение сверки.
        words = page.dedupe_chars().extract_words()
        page.flush_cache()
        page.get_textmap.cache_clear()
    ix, iy = inset_px
    found = set()
    for w in words:
        wx0, wy0, wx1, wy1 = (w["x0"] * scale, w["top"] * scale,
                              w["x1"] * scale, w["bottom"] * scale)
        for (bx0, by0, bx1, by1) in boxes_px:
            if (wx0 >= bx0 + ix and wy0 >= by0 + iy
                    and wx1 <= bx1 - ix and wy1 <= by1 - iy):
                found |= _words(w["text"], min_len)
                break
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
        if not text.strip():
            reason = "регион пуст — рамка обзора могла сместиться"
        elif status == STATUS_MANUAL:
            reason = "нечитаемые места"
        else:
            reason = ""
        pad = cfg["pad_pct"] / 100.0
        inset = (pad * full_img.size[0], pad * full_img.size[1])
        layer = text_layer_words(pdf_path, boxes_px, scale,
                                 cfg["min_word_len"], inset_px=inset)
        mismatch, missing_words = check_against_text_layer(text, layer, cfg)
        if mismatch is not None and mismatch > cfg["text_layer_mismatch_threshold"]:
            status = STATUS_MANUAL
            reason = (f"расхождение с текстовым слоем "
                      f"{mismatch:.0%}: {', '.join(missing_words[:8])}")
        result_regions.append({**_public(region), "text": text, "status": status,
                               "status_reason": reason,
                               "text_layer_mismatch": mismatch})
        if not quiet:
            flag = "⚠" if status == STATUS_MANUAL else "✓"
            print(f"  {flag} {region['id']:4s} {region['kind']:14s} "
                  f"{region['lang']:5s} {len(text)} симв.")

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
        "unread_layer_words": unread[:50],
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
    ap.add_argument("pdf", help="путь к PDF-макету")
    ap.add_argument("-o", "--out", default=None,
                    help="куда сохранить JSON (по умолчанию data/layouts/<имя>.json)")
    args = ap.parse_args(argv)
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


if __name__ == "__main__":
    sys.exit(main())
