# -*- coding: utf-8 -*-
"""Тесты vision-модуля БЕЗ вызовов API: геометрия рамок, разрезание кропов,
сверка с текстовым слоем, чек полноты. Запуск: python tests/test_vision.py"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from labelcheck import render
from labelcheck.vision import (STATUS_MANUAL, STATUS_OK, UNREADABLE_MARK,
                               _words, check_against_text_layer)

CFG = {
    "render_scale": 4, "overview_long_side": 1568, "pad_pct": 4.0,
    "retry_pad_pct": 10.0, "retry_min_side_pct": 14.0,
    "max_crop_px": 2000, "crop_overlap_pct": 10,
    "text_layer_mismatch_threshold": 0.15, "min_word_len": 3,
    "overview_split_aspect": 2.0, "dedup_iou": 0.5,
    "kinds": ["composition", "technical"],
    "required_kinds": ["composition", "nutrition"],
}

PASSED = []


def check(name, cond, note=""):
    if not cond:
        print(f"❌ {name}: {note}")
        sys.exit(1)
    PASSED.append(name)
    print(f"✅ {name}: {note}" if note else f"✅ {name}")


# ── нормализация рамок ───────────────────────────────────────────────────────

def test_normalize_three_scales():
    """Модель отвечает то долями, то процентами, то тысячными —
    нормализация обязана свести все три формата к долям 0..1 (грабли Дня 4)."""
    for raw in ([0.1, 0.2, 0.5, 0.6],       # доли
                [10, 20, 50, 60],           # проценты
                [100, 200, 500, 600]):      # тысячные (формат 0..999)
        box = render.normalize_bbox(raw)
        check("normalize_" + str(raw[0]),
              all(abs(a - b) < 1e-9 for a, b in zip(box, (0.1, 0.2, 0.5, 0.6))),
              f"{raw} → {box}")


def test_normalize_fixes_swapped_and_clamps():
    """Перепутанные углы чинятся, вылезание за края обрезается."""
    box = render.normalize_bbox([500, 600, 100, 200])   # x1<x0, y1<y0
    check("normalize_swapped", box == (0.1, 0.2, 0.5, 0.6), str(box))
    box = render.normalize_bbox([-50, 100, 1200, 900])
    check("normalize_clamp", box[0] == 0.0 and box[2] == 1.0, str(box))


# ── паддинг ──────────────────────────────────────────────────────────────────

def test_pad_bbox():
    """Паддинг 4% расширяет рамку на 0.04 доли с каждой стороны, но не за края."""
    box = render.pad_bbox((0.5, 0.5, 0.6, 0.6), 4.0)
    check("pad_center",
          all(abs(a - b) < 1e-9 for a, b in zip(box, (0.46, 0.46, 0.64, 0.64))),
          str(box))
    box = render.pad_bbox((0.0, 0.01, 0.99, 1.0), 4.0)
    check("pad_clamped", box == (0.0, 0.0, 1.0, 1.0), str(box))


# ── разрезание больших кропов ────────────────────────────────────────────────

def test_split_small_untouched():
    """Кроп в лимите не режется."""
    boxes = render.split_oversized((0, 0, 1500, 1000), 2000, 10)
    check("split_small", boxes == [(0, 0, 1500, 1000)], str(boxes))


def test_split_wide():
    """Широкий кроп режется по вертикали пополам с перехлёстом,
    объединение половинок покрывает исходную рамку."""
    boxes = render.split_oversized((0, 0, 3600, 800), 2000, 10)
    check("split_wide_count", len(boxes) == 2, f"{len(boxes)} частей")
    (l, r) = boxes
    check("split_wide_limit", all(b[2] - b[0] <= 2000 for b in boxes),
          f"ширины {[round(b[2]-b[0]) for b in boxes]}")
    check("split_wide_overlap", l[2] > r[0],
          f"перехлёст {round(l[2]-r[0])} px")
    check("split_wide_cover", l[0] == 0 and r[2] == 3600)


def test_split_tall_recursive():
    """Очень высокий кроп режется рекурсивно, все части в лимите."""
    boxes = render.split_oversized((0, 0, 500, 8000), 2000, 10)
    check("split_tall", all(b[3] - b[1] <= 2000 for b in boxes),
          f"{len(boxes)} частей")


def test_split_never_downscales():
    """Ни одна часть не «ужимается»: каждая часть — фрагмент исходных
    координат (ужатие = галлюцинации, доказано разведкой Дня 4)."""
    src = (100, 200, 4100, 2200)
    for b in render.split_oversized(src, 2000, 10):
        check("split_inside_" + str(round(b[0])),
              b[0] >= src[0] and b[1] >= src[1] and b[2] <= src[2] and b[3] <= src[3])


# ── сверка с текстовым слоем ─────────────────────────────────────────────────

def test_words_extraction():
    """Токенизация сверки: значимые слова, регистр вниз, короткие — вон."""
    w = _words("Состав: капуста, Е621 и e635; на 100 г", 3)
    check("words_basic", "состав" in w and "капуста" in w and "е621" in w, str(w))
    check("words_short_dropped", "на" not in w and "и" not in w)
    check("words_ecode_unified", "е635" in w and "e635" not in w,
          "латинский e635 приведён к кириллице, как в BM25")


def test_tracking_split_word_not_missing():
    """Слой хранит слово кусками («chic»+«ken» — дизайнерский трекинг),
    vision прочитал целиком — подстрочная проверка снимает ложный пропуск."""
    layer = _words("chic ken dumpling", 3)
    ratio, missing = check_against_text_layer("CHICKEN DUMPLING", layer, CFG)
    check("layer_tracking_split", ratio == 0.0, f"missing={missing}")


def test_cid_garbage_dropped():
    """Мусор (cid:NNN) текстового слоя не участвует в сверке."""
    w = _words("(cid:5) (cid:12) состав", 3)
    check("cid_dropped", "cid" not in w and "состав" in w, str(w))


def test_ecode_cross_alphabet_no_false_mismatch():
    """Слой хранит Е-код латиницей, vision прочитал кириллицей —
    расхождения быть не должно."""
    layer = _words("emulsifier E322 stabiliser E621", 3)
    ratio, missing = check_against_text_layer("эмульгатор Е322, Е621", layer, CFG)
    check("layer_ecode_cross", ratio is not None and "е322" not in missing
          and "е621" not in missing, f"missing={missing}")


def test_mismatch_none_without_layer():
    """Макет в кривых (слоя нет) — сверка отвечает None, это не ошибка."""
    ratio, missing = check_against_text_layer("любой текст", None, CFG)
    check("layer_none", ratio is None and missing == [])


def test_mismatch_ratio():
    """Доля пропущенных слов слоя считается честно."""
    layer = {"капуста", "мука", "соль", "сахар"}
    ratio, missing = check_against_text_layer("Состав: капуста, мука, соль", layer, CFG)
    check("layer_ratio", abs(ratio - 0.25) < 1e-9 and missing == ["сахар"],
          f"ratio={ratio}, missing={missing}")


def test_mismatch_zero_when_all_found():
    """Vision может прочитать БОЛЬШЕ слоя (знаки, соседний захват) —
    лишние слова не считаются расхождением."""
    layer = {"капуста", "мука"}
    ratio, _ = check_against_text_layer(
        "Состав: капуста, мука, плюс знак EAC и штрихкод", layer, CFG)
    check("layer_extra_ok", ratio == 0.0)


# ── статусы ──────────────────────────────────────────────────────────────────

def test_expand_to_min():
    """Мелкая рамка растёт от центра до минимума; крупная не меняется;
    у края — обрезается по 0..1."""
    box = render.expand_to_min((0.50, 0.50, 0.54, 0.53), 14.0)
    check("expand_small",
          abs((box[2] - box[0]) - 0.14) < 1e-9 and abs((box[3] - box[1]) - 0.14) < 1e-9
          and abs((box[0] + box[2]) / 2 - 0.52) < 1e-9, str(box))
    big = (0.1, 0.1, 0.5, 0.6)
    check("expand_big_untouched", render.expand_to_min(big, 14.0) == big)
    box = render.expand_to_min((0.0, 0.0, 0.02, 0.02), 14.0)
    check("expand_clamped", box[0] == 0.0 and box[1] == 0.0, str(box))


def test_needs_retry():
    """Повтор с бОльшим паддингом: пусто или сплошное [неразборчиво] — да;
    содержательный текст (даже с отдельными [неразборчиво]) — нет."""
    from labelcheck.vision import _needs_retry
    check("retry_empty", _needs_retry("") and _needs_retry("   "))
    check("retry_unreadable_only",
          _needs_retry("[неразборчиво]") and _needs_retry("[неразборчиво]\n[неразборчиво]"))
    check("retry_not_for_partial",
          not _needs_retry("Состав: капуста [неразборчиво] мука"))


def test_overview_tiles_normal_and_wide():
    """Обычный макет — одна часть; лента шире 2:1 — две половины
    с перехлёстом, вместе покрывающие всю страницу."""
    from PIL import Image
    normal = Image.new("RGB", (1000, 700))
    tiles = render.overview_tiles(normal, 2.0, 10)
    check("tiles_normal", len(tiles) == 1 and tiles[0][1] == (0.0, 0.0, 1.0, 1.0))
    wide = Image.new("RGB", (3000, 1000))
    tiles = render.overview_tiles(wide, 2.0, 10)
    check("tiles_wide_two", len(tiles) == 2)
    (i1, f1), (i2, f2) = tiles
    check("tiles_wide_cover", f1[0] == 0.0 and f2[2] == 1.0 and f1[2] > f2[0],
          f"перехлёст {f1[2]-f2[0]:.2f} доли")
    check("tiles_wide_px", i1.width == round(3000 * f1[2]))


def test_tile_bbox_to_page():
    """Рамка внутри правой половины пересчитывается в координаты страницы."""
    page_box = render.tile_bbox_to_page((0.0, 0.5, 1.0, 1.0),
                                        (0.45, 0.0, 1.0, 1.0))
    check("tile_to_page",
          all(abs(a - b) < 1e-9 for a, b in zip(page_box, (0.45, 0.5, 1.0, 1.0))),
          str(page_box))


def test_bbox_iou_and_dedup():
    """Один блок, увиденный обеими половинами, схлопывается в один регион;
    разные kind не дедупятся."""
    from labelcheck.vision import dedup_regions
    a = {"id": "a", "kind": "marks", "bbox_ok": True, "bbox": (0.40, 0.40, 0.50, 0.50)}
    b = {"id": "b", "kind": "marks", "bbox_ok": True, "bbox": (0.41, 0.41, 0.51, 0.51)}
    c = {"id": "c", "kind": "nutrition", "bbox_ok": True, "bbox": (0.40, 0.40, 0.50, 0.50)}
    check("iou_high", render.bbox_iou(a["bbox"], b["bbox"]) > 0.5)
    check("iou_zero", render.bbox_iou((0, 0, 0.1, 0.1), (0.5, 0.5, 0.6, 0.6)) == 0.0)
    kept = dedup_regions([a, b, c], 0.5)
    check("dedup_keeps", [r["id"] for r in kept] == ["a", "c"],
          str([r["id"] for r in kept]))


def test_missing_kinds_by_marker():
    """Блок с «чужой» меткой, но сигнальными словами в тексте — найден;
    блок без метки и без маркеров — в missing."""
    from labelcheck.vision import missing_kinds
    cfg = {**CFG, "required_kinds": ["composition", "nutrition"],
           "kind_markers": {"composition": ["состав:", "ingredients"],
                            "nutrition": ["ккал", "пищевая ценность"]}}
    regions = [{"kind": "other_text", "text": "RU • Состав: капуста, мука"}]
    check("missing_marker_saves", missing_kinds(regions, cfg) == ["nutrition"],
          str(missing_kinds(regions, cfg)))
    regions.append({"kind": "nutrition", "text": "что угодно"})
    check("missing_kind_saves", missing_kinds(regions, cfg) == [])


def test_unreadable_mark_forces_manual():
    """Правило проекта: [неразборчиво] в тексте → регион на ручную проверку.
    Проверяем константы, на которых логика построена."""
    check("status_consts",
          STATUS_MANUAL == "требует ручной проверки" and STATUS_OK == "прочитано"
          and UNREADABLE_MARK == "[неразборчиво]")


if __name__ == "__main__":
    for name, fn in sorted((k, v) for k, v in globals().items()
                           if k.startswith("test_") and callable(v)):
        fn()
    print(f"\n{len(PASSED)}/{len(PASSED)} проверок пройдено")
