# -*- coding: utf-8 -*-
"""Тесты vision-модуля БЕЗ вызовов API: геометрия рамок, разрезание кропов,
сверка с текстовым слоем, чек полноты. Запуск: python tests/test_vision.py"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from labelcheck import render
from labelcheck import vision as V
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


def test_overview_grid_by_downscale():
    """Сетка обзора по фактору ужатия: лента режется 2x2, обычный макет — цел."""
    from PIL import Image
    big = Image.new("RGB", (390, 410))     # при long_side=100, max_down=2
    tiles = render.overview_grid(big, 100, 2.0, 10)
    assert len(tiles) == 4
    xs = [f for _, f in tiles]
    assert min(x[0] for x in xs) == 0.0 and max(x[2] for x in xs) == 1.0
    assert min(x[1] for x in xs) == 0.0 and max(x[3] for x in xs) == 1.0
    # перехлёст между соседними столбцами есть
    left = sorted(set(round(f[0], 3) for _, f in tiles))
    right = sorted(set(round(f[2], 3) for _, f in tiles))
    assert right[0] > left[1], "нет перехлёста между тайлами"
    small = Image.new("RGB", (150, 180))
    assert len(render.overview_grid(small, 100, 2.0, 10)) == 1
    near = Image.new("RGB", (210, 100))    # перебор 5% — tolerance не плодит тайл
    assert len(render.overview_grid(near, 100, 2.0, 10)) == 1
    PASSED.append("overview_grid")


def test_snap_bbox_expands_to_words():
    """Рамка расширяется до целых слов слоя, но не сжимается и не дальше лимита."""
    words = [("Налейте", (0.10, 0.10, 0.30, 0.14)),   # центр внутри, торчит слева
             ("сосед",   (0.60, 0.50, 0.70, 0.54))]   # центр вне рамки
    got = V.snap_bbox_to_words((0.15, 0.09, 0.40, 0.20), words, 5.0)
    assert abs(got[0] - 0.10) < 1e-9      # расширилась до начала слова
    assert got[2] == 0.40 and got[3] == 0.20  # не сжалась
    got2 = V.snap_bbox_to_words((0.18, 0.09, 0.40, 0.20), words, 5.0)
    assert abs(got2[0] - 0.13) < 1e-9     # лимит 5%: 0.18-0.05, не до 0.10
    PASSED.append("snap_bbox")


def test_ink_ratio_blank_vs_text():
    """Пустая рамка — около нуля, рамка с текстом — заметно больше."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (400, 200), "#ffd24d")
    d = ImageDraw.Draw(img)
    d.rectangle((210, 40, 380, 160), fill="#222222")   # «текст» справа
    assert V.ink_ratio(img, (0.0, 0.0, 0.5, 1.0)) < 0.004
    assert V.ink_ratio(img, (0.5, 0.0, 1.0, 1.0)) > 0.1
    PASSED.append("ink_ratio")


def test_invented_words_detected():
    """Слова vision вне слоя страницы ловятся; числа и короткие — нет."""
    page = [("Importer", None), ("Импортёр", None), ("организация", None)]
    page = [(t, (0, 0, 0, 0)) for t, _ in page]
    fake = V.invented_words("Exporter / Импортёр организация 690033", page)
    assert fake == ["Exporter"]
    assert V.invented_words("Импортёр организация", page) == []
    PASSED.append("invented_words")


def test_merge_regions_glues_adjacent():
    """Примыкающие рамки склеиваются в блок; далёкие и technical — нет."""
    regs = [{"id": "a", "kind": "composition", "bbox": (0.1, 0.1, 0.3, 0.2), "bbox_ok": True},
            {"id": "b", "kind": "usage", "bbox": (0.1, 0.205, 0.3, 0.3), "bbox_ok": True},
            {"id": "c", "kind": "marks", "bbox": (0.7, 0.7, 0.8, 0.8), "bbox_ok": True},
            {"id": "t", "kind": "technical", "bbox": (0.1, 0.21, 0.3, 0.29), "bbox_ok": True}]
    out = V.merge_regions(regs, 0.008)
    ids = [r["id"] for r in out]
    assert "b" not in ids and "a" in ids and "c" in ids and "t" in ids
    merged = next(r for r in out if r["id"] == "a")
    assert merged["bbox"] == (0.1, 0.1, 0.3, 0.3)
    PASSED.append("merge_regions")


def test_layer_fill_covers_orphan_words():
    """Слова слоя вне рамок собираются в новый регион; одиночки — нет."""
    regions = [{"id": "a", "kind": "composition", "bbox": (0.0, 0.0, 0.5, 0.5), "bbox_ok": True}]
    words = [("внутри", (0.1, 0.1, 0.2, 0.12))] +             [(f"с{i}", (0.6, 0.60 + i * 0.012, 0.7, 0.61 + i * 0.012)) for i in range(4)] +             [("одиночка", (0.9, 0.95, 0.95, 0.96))]
    extra = V.layer_fill_regions(regions, words)
    assert len(extra) == 1
    b = extra[0]["bbox"]
    assert 0.59 < b[0] < 0.61 and b[3] > 0.64
    PASSED.append("layer_fill")


def test_invented_ignores_sign_lines_and_spaced_headers():
    """Описания знаков и разреженные заголовки — не «выдумки»."""
    page = [(c, (0, 0, 0, 0)) for c in "П и щ е в а я".split()] +            [("Импортёр", (0, 0, 0, 0)), ("ценность", (0, 0, 0, 0))]
    text = "Пищевая ценность\nзнак EAC с изображением\nИмпортёр Exporter"
    assert V.invented_words(text, page) == ["Exporter"]
    PASSED.append("invented_filters")


# ── R-23: сторож выдумок и текст в кривых ────────────────────────────────────

GUARD_CFG = {**CFG, "invented_min_words": 2, "invented_share": 0.08,
             "layer_partial_ratio": 0.5, "layer_partial_page_share": 0.2}


def _reg(rid, fake, cand, layer_n, vis_n, has_layer=True, status=STATUS_OK):
    return {"id": rid, "status": status, "status_reason": "",
            "invented_words": fake, "invent_candidates": cand,
            "layer_words_in_box": layer_n, "vision_words": vis_n,
            "has_layer": has_layer}


def test_full_layer_region_keeps_invented_guard():
    """Регион целиком в слое, слова вне слоя есть → как раньше, ручная
    проверка (сфабрикованный KR-состав, «Exporter» Дня 8 не теряются)."""
    regs = [_reg("a", ["Exporter", "PENTHIOPIL"], 20, 18, 20),
            _reg("b", [], 30, 30, 30)]
    V.finalize_layer_guards(regs, GUARD_CFG)
    assert regs[0]["status"] == STATUS_MANUAL and "выдумка" in regs[0]["status_reason"]
    assert regs[0]["layer_partial"] is False
    assert regs[1]["status"] == STATUS_OK and regs[1]["status_reason"] == ""
    PASSED.append("guard_full_layer")


def test_partial_region_skips_invented_guard():
    """Регион частично в кривых (слов слоя < половины слов vision: логотип
    SEOUL на манду) — слова вне слоя остаются в списке, статус не трогается."""
    regs = [_reg("logo", ["SEOUL", "ORIGINAL", "서울"], 15, 2, 16),
            _reg("body", [], 100, 95, 100)]
    V.finalize_layer_guards(regs, GUARD_CFG)
    assert regs[0]["status"] == STATUS_OK and regs[0]["layer_partial"] is True
    assert regs[0]["invented_words"] == ["SEOUL", "ORIGINAL", "서울"]
    assert V.page_layer_partial(regs, GUARD_CFG) is False   # 3/115 — страница обычная
    PASSED.append("guard_partial_region")


def test_mixed_pdf_disables_invented_guard_pagewide():
    """PDF со смесью кривых и живого текста (креветки: 41% проверяемых слов
    вне слоя) — сторож выдумок не меняет статус НИ У ОДНОГО региона, даже
    у того, где слов слоя много (t1r2: 24 из 33)."""
    regs = [_reg("t1r2", ["Состав", "Пищевая", "энергетическая", "ценность"], 31, 24, 33),
            _reg("t1r7", ["уполномоченная"] * 16, 18, 2, 23),
            _reg("t2r1", ["Северные"] * 8, 15, 4, 20),
            _reg("mismatch", [], 6, 4, 9, status=STATUS_MANUAL)]
    regs[3]["status_reason"] = "расхождение с текстовым слоем 25%: доля14"
    assert V.page_invented_share(regs) == round(28 / 70, 3)
    assert V.page_layer_partial(regs, GUARD_CFG) is True
    V.finalize_layer_guards(regs, GUARD_CFG)
    assert [r["status"] for r in regs] == [STATUS_OK, STATUS_OK, STATUS_OK, STATUS_MANUAL]
    assert all(r["layer_partial"] for r in regs)
    assert regs[3]["status_reason"].startswith("расхождение")   # сторож слоя жив
    PASSED.append("guard_mixed_pdf")


def test_guard_skips_human_regions_and_no_layer():
    """Регион без слоя — не участвует; регион, подтверждённый человеком, —
    не пересматривается, но в долю по странице входит."""
    regs = [_reg("h", ["Fake", "Word"], 10, 10, 10),
            _reg("n", ["Whatever"] * 5, 5, 0, 5, has_layer=False),
            _reg("ok", [], 40, 40, 40)]
    V.finalize_layer_guards(regs, GUARD_CFG, skip_ids={"h"})
    assert regs[0]["status"] == STATUS_OK          # человек решил — не трогаем
    assert regs[1]["layer_partial"] is False and regs[1]["status"] == STATUS_OK
    assert V.page_invented_share(regs) == round(2 / 50, 3)
    assert V.page_invented_share([regs[1]]) is None
    PASSED.append("guard_skip")


def test_invent_candidates_counts_checked_words_only():
    text = "Состав: мука, соль\nзнак EAC\nштрихкод: 4820140240955\nВода 100 мл"
    assert V.invent_candidates(text) == ["Состав", "мука", "соль", "Вода"]
    PASSED.append("invent_candidates")


def test_reguard_layout_recomputes_without_api():
    """Пересчёт сторожей по готовому layout'у: тексты на месте, статусы
    пересчитаны, правленый человеком регион сохранил статус, но вошёл в
    счётчики страницы. Рендер и слой подменены (без PDF)."""
    from PIL import Image
    layout = {"missing": ["product_name"], "regions": [
        {"id": "r1", "kind": "composition", "lang": "ru", "note": "", "bbox": [0.1, 0.1, 0.5, 0.5],
         "text": "Состав: мука, соль, вода", "status": STATUS_MANUAL,
         "status_reason": "слова вне текстового слоя (возможная выдумка): мука"},
        {"id": "r2", "kind": "other_text", "lang": "ru", "note": "", "bbox": [0.5, 0.5, 0.9, 0.9],
         "text": "Правленый текст человека", "status": STATUS_OK,
         "status_reason": "текст выверен человеком", "human_edited": True},
    ]}
    layout["regions"].append(
        {"id": "r3", "kind": "composition", "lang": "ru", "note": "", "bbox": [0.1, 0.6, 0.5, 0.9],
         "text": "перец черный молотый\nштрихкод: 4820140240955", "status": STATUS_OK,
         "status_reason": ""})
    page = [(w, (0.1, 0.1, 0.2, 0.2)) for w in
            ("Состав", "мука", "соль", "вода", "перец", "черный", "молодой",
             "8", "805957", "025951")]
    layers = {(0.1, 0.1, 0.5, 0.5): {"состав", "мука", "соль", "вода"},
              (0.5, 0.5, 0.9, 0.9): {"состав"},
              (0.1, 0.6, 0.5, 0.9): {"перец", "черный", "молодой"}}
    orig = (render.render_page, V.layer_word_boxes, V.text_layer_words, V.ink_ratio,
            V.page_coverage, render.crop_region)
    render.render_page = lambda pdf, scale: Image.new("RGB", (400, 400), "white")
    render.crop_region = lambda img, bbox, cfg: [(None, tuple(bbox))]
    V.layer_word_boxes = lambda pdf: page
    V.text_layer_words = lambda pdf, boxes, *a, **k: layers[boxes[0]]
    V.ink_ratio = lambda img, bbox: 0.5
    V.page_coverage = lambda pdf, text, cfg: (1.0, [])
    try:
        out = V.reguard_layout(layout, "fake.pdf", GUARD_CFG)
    finally:
        (render.render_page, V.layer_word_boxes, V.text_layer_words, V.ink_ratio,
         V.page_coverage, render.crop_region) = orig
    r1, r2, r3 = out["regions"]
    assert r3["status"] == STATUS_MANUAL and r3["status_reason"].startswith(
        "ВОЗМОЖНАЯ ПОДМЕНА СЛОВА: на макете «молодой», прочитано «молотый»")
    assert r3["word_substitutions"] == [{"layer": "молодой", "vision": "молотый",
                                         "kind": "substitution"}]
    assert r3["invented_digits"] == ["4820140240955"]      # в слое 8805957025951
    assert out["layer_digit_runs"] == ["8805957025951"]
    assert r1["status"] == STATUS_OK and r1["status_reason"] == ""   # всё в слое
    assert r1["invented_words"] == [] and r1["text"] == "Состав: мука, соль, вода"
    assert r2["status"] == STATUS_OK and r2["status_reason"] == "текст выверен человеком"
    assert r2["invented_words"] == ["Правленый", "текст", "человека"]  # посчитано, статус цел
    # доля по странице считается с правленым регионом: 3 его слова + «молотый»
    # вне слоя из 10 проверяемых — 0.4 ≥ 0.2, страница «частично в кривых»
    assert out["text_layer_partial"] is True and out["text_layer_invented_share"] == 0.4
    assert out["text_layer_coverage"] == 1.0
    # missing[] пересчитан по конфигу (R-41): старое «product_name» ушло,
    # а отсутствующие по сигнальным словам типы — на месте
    assert "product_name" not in out["missing"] and "nutrition" in out["missing"]
    PASSED.append("reguard")


# ── R-15 / R-16: подмены слов, цифры, стык половин кропа ─────────────────────

def test_layer_core_is_union_of_crop_pieces():
    """Слово на стыке двух половин длинного кропа (в кайме каждой, но внутри
    объединения) участвует в сверке; слово во внешней кайме — нет."""
    def w(text, x0, x1, y0=10, y1=20):
        return {"text": text, "x0": x0, "x1": x1, "top": y0, "bottom": y1}
    pieces = [(0, 0, 1000, 100), (900, 0, 1900, 100)]     # перехлёст 900..1000
    words = [w("стык", 940, 990), w("край", 5, 60), w("центр", 300, 400),
             w("правый", 1500, 1600)]
    got = V.layer_words_in_core(words, pieces, scale=1, min_len=3, inset_px=(100, 5))
    assert got == {"стык", "центр", "правый"}, got
    assert V.layer_words_in_core([], pieces, 1, 3) is None
    PASSED.append("layer_core_union")


def test_digit_runs_and_layer_confirmation():
    """Числа ≥ 8 цифр (пробелы внутри снимаются); число vision, которого нет
    в числах слоя, — «выдумка»; без чисел в слое сравнивать не с чем."""
    assert V.digit_runs("EAN 8 805957 025951, тел. 123, партия 2026") == ["8805957025951"]
    page = [(t, (0, 0, 0, 0)) for t in "штрихкод 8 805957 025951 масса 240".split()]
    runs = V.layer_digit_runs(page)
    assert runs == ["8805957025951"], runs
    assert V.invented_digits("штрихкод: 4820140240955", runs) == ["4820140240955"]
    assert V.invented_digits("штрихкод: 8805957025951", runs) == []
    assert V.invented_digits("штрихкод: 4820140240955", []) == []   # слой без цифр
    assert V.layer_digit_runs(None) == []
    PASSED.append("digit_runs")


def test_levenshtein_and_word_substitutions():
    """«молодой» → «молотый» — подмена (расстояние 2: д→т, о→ы); «hалейте» →
    «Налейте» — гомоглиф; короткие слова и далёкие пары («перец»/«перчик»,
    расстояние 3) не считаются; каждое слово vision участвует один раз."""
    assert V.levenshtein("молодой", "молотый") == 2
    assert V.levenshtein("перец", "перчик") == 3
    assert V.levenshtein("кот", "слон", 2) == 3          # ранний выход: > limit
    cfg = {"substitution_max_distance": 2, "substitution_min_len": 5}
    pairs = V.word_substitutions(
        ["молодой", "hалейте", "соль", "перец", "варить"],
        ["молотый", "Налейте", "сель", "перчик", "жарить"], cfg)
    assert pairs == [{"layer": "молодой", "vision": "молотый", "kind": "substitution"},
                     {"layer": "hалейте", "vision": "Налейте", "kind": "homoglyph"},
                     {"layer": "варить", "vision": "жарить", "kind": "substitution"}], pairs
    # «соль» короче 5 букв — не пара, хотя «сель» рядом
    assert V.word_substitutions(["молодой", "молодая"], ["молотый"], cfg) == [
        {"layer": "молодой", "vision": "молотый", "kind": "substitution"}]
    assert V.word_substitutions([], ["молотый"], cfg) == []
    PASSED.append("word_substitutions")


def test_required_kinds_without_product_name():
    """R-10: название не в required_kinds — у него нет сигнальных слов, и
    проверка держалась на недетерминированной kind-метке."""
    cfg = render.load_config(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "labelcheck", "config.yaml"))
    assert "product_name" not in cfg["required_kinds"]
    assert "composition" in cfg["required_kinds"]
    regions = [{"kind": "other_text", "text": "Лепешка Roti. Состав: мука. Масса нетто 400 г. "
                                              "Пищевая ценность. Годен до. Изготовитель"}]
    assert V.missing_kinds(regions, cfg) == []
    PASSED.append("required_kinds")


if __name__ == "__main__":
    for name, fn in sorted((k, v) for k, v in globals().items()
                           if k.startswith("test_") and callable(v)):
        fn()
    print(f"\n{len(PASSED)}/{len(PASSED)} проверок пройдено")
