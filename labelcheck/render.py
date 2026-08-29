# -*- coding: utf-8 -*-
"""Рендер PDF-макета упаковки в изображения для vision-чтения.

Три задачи (параметры — labelcheck/config.yaml, секция vision):
1. render_page()    — страница PDF → полноразмерный PNG (scale 4: кегль 6 pt → 24 px).
2. make_overview()  — уменьшенная копия для обзорного прохода (карта регионов).
3. crop_region()    — вырез региона из полноразмерного рендера по рамке обзора:
                      паддинг → обрезка по краям → при превышении max_crop_px
                      разрез пополам с перехлёстом (НЕ даунскейл: ужатие
                      мелкого шрифта даёт галлюцинации — замер Дня 4).

Координаты рамок всюду — доли стороны изображения 0..1 (после normalize_bbox).
"""

from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image


def load_config(path="labelcheck/config.yaml"):
    """Читает секцию vision из YAML-конфига."""
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["vision"]


def render_page(pdf_path, scale, page_index=0):
    """Страница PDF → PIL-изображение в полном разрешении.

    scale — множитель от 72 dpi: пиксели = пункты × scale.
    Макет — одна страница; page_index на будущее (многостраничные PDF).
    """
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        page = pdf[page_index]
        bitmap = page.render(scale=scale)
        return bitmap.to_pil()
    finally:
        pdf.close()


def page_count(pdf_path):
    """Число страниц в PDF (макет обычно одностраничный)."""
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        return len(pdf)
    finally:
        pdf.close()


def make_overview(full_img, long_side):
    """Уменьшенная копия полноразмерного рендера для обзорного прохода.

    Долгая сторона — long_side px; если изображение и так меньше,
    возвращается как есть (увеличивать смысла нет).
    """
    w, h = full_img.size
    k = long_side / max(w, h)
    if k >= 1:
        return full_img
    return full_img.resize((round(w * k), round(h * k)), Image.LANCZOS)


def overview_tiles(full_img, max_aspect, overlap_pct):
    """Части полноразмерного рендера для обзора: 1 часть для обычного макета,
    2 половины с перехлёстом — для сильно вытянутого (лента-развёртка).

    На обзоре целой ленты (аспект > 2) мелкие блоки занимают единицы
    пикселей — обзор их пропускает целиком (пропуск блока на тестовой ленте-развёртке, День 4).
    Возвращает список (изображение, рамка части в долях страницы 0..1).
    """
    W, H = full_img.size
    aspect = max(W, H) / min(W, H)
    if aspect <= max_aspect:
        return [(full_img, (0.0, 0.0, 1.0, 1.0))]
    ov = overlap_pct / 100.0
    if W >= H:
        return [
            (full_img.crop((0, 0, round(W * (0.5 + ov / 2)), H)),
             (0.0, 0.0, 0.5 + ov / 2, 1.0)),
            (full_img.crop((round(W * (0.5 - ov / 2)), 0, W, H)),
             (0.5 - ov / 2, 0.0, 1.0, 1.0)),
        ]
    return [
        (full_img.crop((0, 0, W, round(H * (0.5 + ov / 2)))),
         (0.0, 0.0, 1.0, 0.5 + ov / 2)),
        (full_img.crop((0, round(H * (0.5 - ov / 2)), W, H)),
         (0.0, 0.5 - ov / 2, 1.0, 1.0)),
    ]


def tile_bbox_to_page(bbox, tile_frame):
    """Рамка 0..1 внутри части обзора → рамка 0..1 всей страницы."""
    tx0, ty0, tx1, ty1 = tile_frame
    w, h = tx1 - tx0, ty1 - ty0
    x0, y0, x1, y1 = bbox
    return (tx0 + x0 * w, ty0 + y0 * h, tx0 + x1 * w, ty0 + y1 * h)


def bbox_iou(a, b):
    """Пересечение-над-объединением двух рамок 0..1 (дедуп регионов
    из зоны перехлёста половинок обзора)."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix0 >= ix1 or iy0 >= iy1:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    union = area(a) + area(b) - inter
    return inter / union if union else 0.0


def normalize_bbox(raw_bbox):
    """Рамка из ответа модели → доли 0..1.

    Модель просят отвечать целыми 0..999 (её «родной» формат), но на
    практике формат гуляет: проценты 0..100, доли 0..1, тысячные 0..999.
    Определяем масштаб по максимальному значению — иначе рамки «уезжают»
    в 10 раз (наблюдалось в разведке Дня 4).
    """
    x0, y0, x1, y1 = (float(v) for v in raw_bbox)
    m = max(abs(x0), abs(y0), abs(x1), abs(y1))
    if m <= 1.0:
        div = 1.0
    elif m <= 100.0:
        div = 100.0
    else:
        div = 1000.0
    box = [x0 / div, y0 / div, x1 / div, y1 / div]
    # перепутанные углы и вылезание за края — чиним молча, это не ошибка данных
    x0, x1 = sorted((box[0], box[2]))
    y0, y1 = sorted((box[1], box[3]))
    clamp = lambda v: min(1.0, max(0.0, v))
    return (clamp(x0), clamp(y0), clamp(x1), clamp(y1))


def pad_bbox(bbox, pad_pct):
    """Расширяет рамку на pad_pct (доли процента стороны макета) с обрезкой 0..1.

    Рамки обзора гуляют на 2–4% (замер) — без запаса край блока обрезается.
    """
    pad = pad_pct / 100.0
    x0, y0, x1, y1 = bbox
    return (max(0.0, x0 - pad), max(0.0, y0 - pad),
            min(1.0, x1 + pad), min(1.0, y1 + pad))


def expand_to_min(bbox, min_side_pct):
    """Расширяет рамку 0..1 симметрично от центра до минимум min_side_pct стороны.

    Для повторного чтения мелких регионов: сдвиг рамки обзора нередко больше
    самого блока, и паддинг не спасает — а зона в 12–15% стороны страницы
    вокруг центра почти всегда накрывает уехавший блок.
    """
    mn = min_side_pct / 100.0
    x0, y0, x1, y1 = bbox
    if x1 - x0 >= mn and y1 - y0 >= mn:
        return bbox  # обе стороны уже не меньше минимума — не трогаем
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w = max(x1 - x0, mn)
    h = max(y1 - y0, mn)
    clamp = lambda v: min(1.0, max(0.0, v))
    return (clamp(cx - w / 2), clamp(cy - h / 2),
            clamp(cx + w / 2), clamp(cy + h / 2))


def split_oversized(box_px, max_px, overlap_pct):
    """Пиксельная рамка → список рамок, каждая не длиннее max_px.

    Кроп длиннее max_px нельзя отдавать модели: она молча ужмёт его до
    своего окна, и мелкий шрифт станет нечитаемым (галлюцинации — замер
    Дня 4). Поэтому режем пополам вдоль длинной стороны с перехлёстом,
    рекурсивно, пока обе стороны не в лимите. Перехлёст гарантирует, что
    строка на линии разреза целиком видна хотя бы в одной половине.
    """
    x0, y0, x1, y1 = box_px
    w, h = x1 - x0, y1 - y0
    if max(w, h) <= max_px:
        return [box_px]
    overlap = max(w, h) * overlap_pct / 100.0
    if w >= h:  # режем по вертикали (широкий кроп)
        mid = x0 + w / 2
        left = (x0, y0, mid + overlap / 2, y1)
        right = (mid - overlap / 2, y0, x1, y1)
        return split_oversized(left, max_px, overlap_pct) + \
            split_oversized(right, max_px, overlap_pct)
    mid = y0 + h / 2  # режем по горизонтали (высокий кроп)
    top = (x0, y0, x1, mid + overlap / 2)
    bottom = (x0, mid - overlap / 2, x1, y1)
    return split_oversized(top, max_px, overlap_pct) + \
        split_oversized(bottom, max_px, overlap_pct)


def crop_region(full_img, bbox, cfg):
    """Регион макета → список PIL-кропов полного разрешения (обычно один).

    bbox — нормализованная рамка 0..1 БЕЗ паддинга (паддинг добавляется здесь,
    чтобы тесты и вызывающий код работали с «сырой» рамкой обзора).
    Возвращает список пар (кроп, его пиксельная рамка в полном рендере) —
    рамка нужна для сверки с текстовым слоем PDF.
    """
    W, H = full_img.size
    x0, y0, x1, y1 = pad_bbox(bbox, cfg["pad_pct"])
    box_px = (x0 * W, y0 * H, x1 * W, y1 * H)
    pieces = split_oversized(box_px, cfg["max_crop_px"], cfg["crop_overlap_pct"])
    out = []
    for p in pieces:
        p_int = tuple(round(v) for v in p)
        if p_int[2] - p_int[0] < 8 or p_int[3] - p_int[1] < 8:
            continue  # вырожденный обрезок — пропускаем
        out.append((full_img.crop(p_int), p_int))
    return out


def save_png(img, path):
    """Сохраняет изображение, создавая папку при необходимости."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    return path
