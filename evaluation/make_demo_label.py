"""Синтетический демо-макет упаковки для ревьюеров (День 10).

Реальные макеты компании в репозиторий не попадают (ТЗ §0), а без макета
ревьюер не сможет пройти сценарий шаг 1 → шаг 3. Скрипт рисует вымышленную
этикетку «Ягодная смесь «Лесная поляна»» (замороженные ягоды — профиль
компании, без мяса) и сохраняет как РАСТРОВЫЙ PDF без текстового слоя —
как макет «в кривых», самый частый случай препресса; система читает его
vision-моделью. На макете намеренно оставлены дефекты, чтобы было что
находить: Е-код латинской буквой (E330), «мороженная» вместо «мороженая»;
дата — шаблоном DD.MM.YYYY (норма препресса, не дефект).

Запуск из корня:  python evaluation/make_demo_label.py
Результат: data/samples/demo_label.pdf (в git). Шрифт — DejaVu Sans
(есть в большинстве Linux-дистрибутивов и в образе python:3.12-slim после
установки fonts-dejavu; на маке подставится /Library/Fonts/Arial Unicode).
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "samples" / "demo_label.pdf"

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if bold and "Bold" not in path and "DejaVuSans.ttf" in path:
            p = path.replace("DejaVuSans.ttf", "DejaVuSans-Bold.ttf")
            if Path(p).exists():
                return ImageFont.truetype(p, size)
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise SystemExit("нет TTF-шрифта с кириллицей: установите fonts-dejavu")


def ean13(prefix12: str) -> str:
    """Контрольная цифра EAN-13 к 12 цифрам."""
    s = sum(int(d) * (3 if i % 2 else 1) for i, d in enumerate(prefix12))
    return prefix12 + str((10 - s % 10) % 10)


def draw_front(d: ImageDraw.ImageDraw, x0: int, y0: int, w: int, h: int) -> None:
    d.rectangle((x0, y0, x0 + w, y0 + h), fill=(238, 246, 233), outline=(60, 90, 60), width=3)
    d.text((x0 + 40, y0 + 40), "ЛЕСНАЯ ПОЛЯНА", font=font(64, True), fill=(30, 70, 40))
    d.text((x0 + 40, y0 + 125), "Ягодная смесь быстрозамороженная", font=font(34), fill=(30, 30, 30))
    d.text((x0 + 40, y0 + 170), "черника · малина · ежевика · клубника", font=font(28), fill=(80, 80, 80))
    d.text((x0 + 40, y0 + 300), "FROZEN BERRY MIX", font=font(30), fill=(60, 60, 60))
    d.text((x0 + 40, y0 + h - 130), "Масса нетто 300 г", font=font(40, True), fill=(30, 30, 30))
    d.text((x0 + 40, y0 + h - 75), "Net weight 300 g", font=font(26), fill=(80, 80, 80))
    # ДЕМО-пометка — чтобы макет нельзя было принять за реальный
    d.text((x0 + w - 330, y0 + 30), "ДЕМО-МАКЕТ", font=font(26, True), fill=(180, 40, 40))
    d.text((x0 + w - 330, y0 + 62), "вымышленный продукт", font=font(20), fill=(180, 40, 40))


def draw_back(d: ImageDraw.ImageDraw, x0: int, y0: int, w: int, h: int) -> None:
    d.rectangle((x0, y0, x0 + w, y0 + h), fill=(255, 255, 255), outline=(60, 90, 60), width=3)
    f, fb, fs = font(22), font(22, True), font(19)
    y = y0 + 30
    lines = [
        (fb, "Ягодная смесь быстрозамороженная «Лесная поляна»"),
        (f, "Состав: черника, малина, ежевика, клубника, регулятор кислотности E330."),
        (f, "Может содержать следы орехов. Продукт не содержит ГМО."),
        (f, ""),
        (fb, "Пищевая ценность на 100 г продукта:"),
        (f, "белки 1,0 г · жиры 0,5 г · углеводы 10,0 г · энергетическая ценность 48 ккал / 200 кДж"),
        (f, ""),
        (f, "Условия хранения: при температуре не выше минус 18 °C."),
        (f, "Срок годности: 24 месяца. После размораживания повторно не замораживать."),
        (f, "Дата изготовления: DD.MM.YYYY · Годен до: DD.MM.YYYY"),
        (f, "Способ приготовления: разморозить при комнатной температуре 20–30 минут."),
        (f, "Ягода мороженная, промыть перед употреблением."),
        (f, ""),
        (f, "Изготовитель: ООО «Пример-Фрост», 123456, Россия, г. Примерск, ул. Ягодная, д. 1."),
        (f, "Импортёр / организация, уполномоченная на принятие претензий: ООО «Пример-Фрост»."),
        (f, "Тел.: +7 (000) 000-00-00 · info@example.com"),
    ]
    for fnt, text in lines:
        d.text((x0 + 30, y), text, font=fnt, fill=(20, 20, 20))
        y += 34 if text else 16
    # Знак EAC (рисуем буквами в рамке) и штрихкод
    bx, by = x0 + w - 150, y0 + h - 150
    d.rectangle((bx, by, bx + 100, by + 70), outline=(0, 0, 0), width=4)
    d.text((bx + 12, by + 14), "EAC", font=font(36, True), fill=(0, 0, 0))
    code = ean13("460000000001")
    cx, cy = x0 + 30, y0 + h - 130
    for i, ch in enumerate(code * 3):
        wbar = 2 + int(ch) % 3
        if i % 2 == 0:
            d.rectangle((cx, cy, cx + wbar, cy + 70), fill=(0, 0, 0))
        cx += wbar + 2
    d.text((x0 + 30, cy + 78), " ".join((code[0], code[1:7], code[7:])), font=fs, fill=(0, 0, 0))
    d.text((x0 + w - 420, y0 + h - 40), "05 PP · знак «для пищевой продукции»", font=fs, fill=(60, 60, 60))


def build(out: Path = OUT, dpi: int = 200) -> Path:
    W, H = int(297 / 25.4 * dpi), int(148 / 25.4 * dpi)   # развёртка 297×148 мм
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    gap = 40
    panel_w = (W - 3 * gap) // 2
    draw_front(d, gap, gap, panel_w, H - 2 * gap)
    draw_back(d, 2 * gap + panel_w, gap, panel_w, H - 2 * gap)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "PDF", resolution=dpi)
    return out


if __name__ == "__main__":
    p = build()
    print(f"демо-макет: {p} ({p.stat().st_size // 1024} КБ)")
    sys.exit(0)
