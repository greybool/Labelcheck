"""Streamlit UI LabelCheck (блок C, день 9). Запуск с мака:

    streamlit run labelcheck/app.py

Три шага, между ними — кнопки перехода (навигация состоянием, не вкладками:
вкладки Streamlit нельзя переключить из кода, а «Далее» должна работать):

1. «Макет» — PDF → vision → просмотр с масштабом и выбором блоков кликом,
   правка текста ЧЕЛОВЕКОМ до вердиктов (правки в layout-JSON).
2. «Проверка» — выбор профильных регламентов → вердикты по 21 аспекту →
   по каждому замечанию: решение «что делать» и оценка ответа системы,
   сохраняются САМИ (кнопок «сохранить» нет).
3. «План работ» — короткие списки без цитат регламентов: дизайнеру,
   поставщику, проверить самому; копирование и выгрузка в Markdown/Word.
4. «Мониторинг» — графики по журналу проверок и оценкам (День 10;
   данные — labelcheck/dashboard.py, графики — labelcheck/dashboard_ui.py).
"""

import html
import json
import sys
import time
from pathlib import Path

# streamlit run labelcheck/app.py исполняет файл как скрипт — корень проекта
# в sys.path не попадает, и «import labelcheck» падает. Добавляем корень сами.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from labelcheck.actions import (TARGETS, apply_human_decisions, build_plan,
                                plan_to_docx, render_plan_markdown)
from labelcheck import dashboard_ui
from labelcheck.retrieval import ROOT, Retriever, load_config
from labelcheck.store import (apply_region_edit, confirm_region, connect,
                              decisions_for_check, fetch_checks,
                              fetch_feedback, record_check, save_layout,
                              upsert_feedback)
from labelcheck.verdict import (STATUS_COMPLIANT, STATUS_MANUAL,
                                STATUS_VIOLATION, check_layout,
                                render_markdown)

st.set_page_config(page_title="LabelCheck", page_icon="🏷️", layout="wide")

CFG = load_config()
LAYOUTS_DIR = ROOT / "data" / "layouts"
UPLOADS_DIR = ROOT / CFG["ui"]["uploads_dir"]
UI_REPORTS_DIR = ROOT / CFG["ui"]["reports_dir"]

ICONS = {STATUS_VIOLATION: "🔴", STATUS_MANUAL: "🟡", STATUS_COMPLIANT: "🟢"}
REGION_COLORS = {"прочитано": (46, 160, 67), "требует ручной проверки": (219, 154, 4)}

STEPS = ["1 · Макет", "2 · Проверка", "3 · План работ", "4 · Мониторинг"]
ZOOM_STEPS = [50, 75, 100, 150, 200, 300]

CATEGORY_LABELS = {"meat": "🥩 Мясная продукция",
                   "poultry": "🍗 Продукция из мяса птицы",
                   "fish": "🐟 Рыба и морепродукты"}
CATEGORY_HINTS = {"meat": "ТР ТС 034/2013 — мясо и мясная продукция",
                  "poultry": "ТР ЕАЭС 051/2021 — мясо птицы",
                  "fish": "ТР ЕАЭС 040/2016 — рыба и морепродукты"}
# Решение по замечанию: ключ → (подпись, пояснение)
DECISIONS = {"none": ("Ничего не требуется", "замечание снято"),
             "designer": ("Замечание дизайнеру", "поправить в макете"),
             "supplier": ("Запросить у поставщика", "нужны данные производителя"),
             "manual": ("Проверить самому", "смотрю вручную")}
RATINGS = ["не оценивал", "👍 верно", "👎 система ошиблась"]


# ── кэшируемые тяжёлые объекты ───────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_retriever_and_client():
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(ROOT / ".env")
    client = OpenAI()
    return Retriever(CFG, openai_client=client), client


@st.cache_data(show_spinner=False)
def rendered_page(pdf_path: str, mtime: float):
    from labelcheck import render
    return render.render_page(pdf_path, scale=2)


@st.cache_data(show_spinner=False)
def overlay_image(pdf_path: str, mtime: float, regions_key: str,
                  selected_id: str | None, width: int):
    """Рендер с рамками и подписями, отмасштабированный под ширину просмотра.
    Кэш по ключу (макет, набор рамок, выбранный блок, ширина) — перелистывание
    блоков и зум не пересобирают картинку заново."""
    regions = json.loads(regions_key)
    base = rendered_page(pdf_path, mtime)
    img = base.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    w, h = img.size
    size = max(14, int(min(w, h) * 0.018))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    for r in regions:
        bbox = r.get("bbox")
        if not bbox:
            continue
        x0, y0, x1, y1 = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h
        chosen = r["id"] == selected_id
        color = ((31, 111, 235) if chosen
                 else REGION_COLORS.get(r.get("status"), (110, 110, 110)))
        d.rectangle([x0, y0, x1, y1], outline=color, width=6 if chosen else 3)
        tag = f"{r['id']} · {r['kind']}"
        tw = d.textlength(tag, font=font)
        ty = max(0, y0 - size - 6)
        d.rectangle([x0, ty, x0 + tw + 8, ty + size + 6], fill=color)
        d.text((x0 + 4, ty + 3), tag, fill=(255, 255, 255), font=font)
    return img.resize((width, max(1, int(img.height * width / img.width))),
                      Image.LANCZOS)


@st.cache_data(show_spinner=False)
def region_crop(pdf_path: str, mtime: float, bbox_json: str, width: int):
    """Увеличенный фрагмент вокруг выбранного блока (решение Сергея вместо
    перетаскивания мышью: практическая задача — рассмотреть мелкий текст).
    Запас вокруг рамки — доля стороны страницы плюс доля размера самого
    блока (config → ui.crop_pad_page_pct / crop_pad_block_pct): рамки
    обзора гуляют на 2–4% страницы, и у макетов без текстового слоя
    (Roti, R-35) их нечем прищёлкнуть — блок показывался обрезанным."""
    bbox = json.loads(bbox_json)
    if not bbox:
        return None
    base = rendered_page(pdf_path, mtime)
    w, h = base.size
    pp = CFG["ui"].get("crop_pad_page_pct", 4) / 100.0
    pb = CFG["ui"].get("crop_pad_block_pct", 10) / 100.0
    pad_x = pp * w + pb * (bbox[2] - bbox[0]) * w
    pad_y = pp * h + pb * (bbox[3] - bbox[1]) * h
    box = (max(0, bbox[0] * w - pad_x), max(0, bbox[1] * h - pad_y),
           min(w, bbox[2] * w + pad_x), min(h, bbox[3] * h + pad_y))
    crop = base.crop(tuple(int(v) for v in box))
    if crop.width < 5 or crop.height < 5:
        return None
    scale = max(1.0, width / crop.width)
    return crop.resize((int(crop.width * scale), int(crop.height * scale)),
                       Image.LANCZOS)


def load_click_component():
    """Компонент кликов по картинке. Возвращает функцию или None, положив
    причину в session_state: «пакет установлен, а приложение говорит, что
    нет» бывает при установке в другую среду или смене API — пользователю
    нужна настоящая причина, а не общая фраза."""
    try:
        import streamlit_image_coordinates as m
    except Exception as e:  # noqa: BLE001
        st.session_state["click_component_error"] = f"{type(e).__name__}: {e}"
        return None
    for attr in ("streamlit_image_coordinates", "image_coordinates", "main"):
        fn = getattr(m, attr, None)
        if callable(fn):
            return fn
    st.session_state["click_component_error"] = (
        f"в пакете {getattr(m, '__file__', '?')} нет функции "
        "streamlit_image_coordinates")
    return None


TOP_ANCHOR_ID = "labelcheck-top"


def top_anchor():
    """Невидимая метка в самом верху страницы — цель прокрутки при смене
    шага. Якорь надёжнее, чем прокрутка контейнера: не нужно угадывать,
    какой именно элемент Streamlit прокручивает в текущей версии."""
    st.markdown(f"<div id='{TOP_ANCHOR_ID}'></div>", unsafe_allow_html=True)


def scroll_to_top():
    """Прокрутка к метке верха после смены шага.

    Streamlit сохраняет позицию прокрутки при перерисовке, и пользователь
    попадает в середину нового шага. Прокручиваем к якорю через
    scrollIntoView (работает независимо от того, скроллится документ или
    внутренний контейнер) и повторяем, пока страница дорисовывается:
    на длинном третьем шаге вёрстка завершается позже первой попытки."""
    st.iframe(
        f"""<script>
        const ID = "{TOP_ANCHOR_ID}";
        function up() {{
          try {{
            const doc = window.parent.document;
            const el = doc.getElementById(ID);
            if (el) el.scrollIntoView({{block: "start", behavior: "instant"}});
            (doc.scrollingElement || doc.documentElement).scrollTop = 0;
            doc.querySelectorAll('section.main, div[data-testid="stMain"], '
              + 'div[data-testid="stAppViewContainer"], '
              + 'div[data-testid="stMainBlockContainer"]')
              .forEach(n => {{ try {{ n.scrollTop = 0; }} catch (e) {{}} }});
            window.parent.scrollTo(0, 0);
          }} catch (e) {{ /* песочница закрыла доступ — молча выходим */ }}
        }}
        [0, 50, 150, 300, 600, 1000, 1600, 2400].forEach(ms => setTimeout(up, ms));
        </script>""",
        height=1)   # 0 недопустим: Streamlit требует положительную высоту


def region_at(regions, x_rel, y_rel):
    """Блок под кликом (самый маленький из попавших — вложенные рамки)."""
    hits = [r for r in regions
            if r.get("bbox") and r["bbox"][0] <= x_rel <= r["bbox"][2]
            and r["bbox"][1] <= y_rel <= r["bbox"][3]]
    if not hits:
        return None
    return min(hits, key=lambda r: ((r["bbox"][2] - r["bbox"][0]) *
                                    (r["bbox"][3] - r["bbox"][1])))["id"]


def find_source_pdf(layout: dict) -> Path | None:
    name = (layout.get("meta") or {}).get("source_pdf")
    if not name:
        return None
    for folder in (ROOT / "data" / "samples_private", UPLOADS_DIR):
        p = folder / name
        if p.exists():
            return p
    return None


def quote_block(text: str, address: str):
    """Цитата пункта регламента через HTML: текст пункта часто начинается
    с «3. …», и markdown превращал его в нумерованный список, ломая
    отступы и цвет (замечание Сергея 31.08)."""
    body = html.escape(text).replace("\n", "<br>")
    st.markdown(
        f"<div style='border-left:3px solid #9aa0a6;padding:6px 12px;"
        f"margin:6px 0;background:rgba(150,150,150,.08);'>"
        f"<div style='opacity:.75;font-size:.85em;margin-bottom:4px'>"
        f"📖 {html.escape(address)}</div>{body}</div>",
        unsafe_allow_html=True)


def copy_button(text: str, label: str = "📋 Скопировать текст"):
    """Всегда видимая кнопка копирования (встроенная в st.code появляется
    только при наведении). Работает через буфер обмена браузера с запасным
    вариантом execCommand для строгих настроек."""
    payload = json.dumps(text)
    st.iframe(
        f"""
        <button id="cp" style="width:100%;padding:.55rem 1rem;font-size:15px;
            border-radius:.5rem;border:1px solid rgba(120,120,120,.5);
            background:#fff;cursor:pointer;">{label}</button>
        <script>
        const txt = {payload};
        const btn = document.getElementById('cp');
        btn.onclick = async () => {{
          try {{ await navigator.clipboard.writeText(txt); }}
          catch (e) {{
            const ta = document.createElement('textarea');
            ta.value = txt; document.body.appendChild(ta); ta.select();
            document.execCommand('copy'); ta.remove();
          }}
          btn.textContent = '✅ Скопировано';
          setTimeout(() => btn.textContent = {json.dumps(label)}, 1800);
        }};
        </script>
        """, height=52)


def layer_diff_table(region: dict) -> None:
    """Полная сверка блока с текстовым слоем PDF (REVIEW-LOG R-12): подмены
    слов, слова слоя, которых нет в прочитанном тексте, и слова прочитанного
    текста, которых нет в слое. Раньше в подписи блока показывались 8 слов
    из ~50 — человек не мог понять, что именно расходится."""
    pairs = region.get("word_substitutions") or []
    missing = region.get("layer_missing_words") or []
    paired_layer = {p["layer"] for p in pairs}
    paired_vision = {p["vision"] for p in pairs}
    invented = [w for w in (region.get("invented_words") or []) if w not in paired_vision]
    digits = region.get("invented_digits") or []
    missing = [w for w in missing if w not in paired_layer]
    if not (pairs or missing or invented or digits):
        if region.get("has_layer"):
            st.caption("Сверка с текстовым слоем: расхождений нет.")
        return
    total = len(pairs) + len(missing) + len(invented) + len(digits)
    with st.expander(f"Сверка с текстовым слоем PDF — расхождений: {total}",
                     expanded=True):
        if pairs:
            st.markdown("**Возможные подмены слов** — vision прочитал не то, "
                        "что напечатано; проверьте макет:")
            st.table([{"на макете (слой PDF)": p["layer"], "прочитано": p["vision"],
                       "тип": ("латиница вместо кириллицы" if p["kind"] == "homoglyph"
                               else "другое слово")} for p in pairs])
        if region.get("layer_partial"):
            st.caption("Блок частично в кривых: слова «в слое нет» здесь — скорее "
                       "всего, текст в кривых, а не выдумка.")
        c1, c2 = st.columns(2)
        c1.markdown(f"**В текстовом слое PDF есть, но при распознавании макета "
                    f"не прочитано ({len(missing)})**")
        c1.markdown(", ".join(missing) if missing else "—")
        c2.markdown(f"**Прочитано при распознавании, но в текстовом слое PDF "
                    f"нет ({len(invented) + len(digits)})**")
        c2.markdown(", ".join(invented + digits) if (invented or digits) else "—")


def human_summary(meta: dict) -> str:
    """Сводка прогона человеческими словами. Профильные регламенты в
    интерфейсе выбирает только человек (решение Сергея 02.09, R-05) —
    автоопределение в тексте не упоминается; старые отчёты со scan
    targeted/full описываются нейтрально."""
    cats = meta.get("categories") or {}
    if cats:
        names = ", ".join(CATEGORY_LABELS.get(c, c).split(" ", 1)[-1] for c in cats)
        scan = ("выбраны вручную" if meta.get("category_scan") == "manual"
                else "по маркерам в тексте — старый режим")
        cat_txt = f"Профильные регламенты: {names} ({scan})"
    else:
        cat_txt = ("Профильные регламенты не подключались — проверка по общим "
                   "правилам маркировки (022, 021, 029, 005)")
    sec = meta.get("seconds") or 0
    dur = (f"{int(sec // 60)} мин {int(sec % 60)} с" if sec >= 60 else f"{int(sec)} с")
    return f"{cat_txt}. Проверка заняла {dur}."


# ── состояние ────────────────────────────────────────────────────────────────

ss = st.session_state
for key, default in (("layout", None), ("layout_path", None), ("report", None),
                     ("check_id", None), ("plan", None), ("sel_region", None),
                     ("nav", STEPS[0]), ("zoom", 100)):
    ss.setdefault(key, default)


def go_to(step_name: str):
    ss.nav = step_name
    ss.scroll_top = True   # новый шаг открывается сверху, а не с середины


def save_decision(aspect_id: int, aspect_name: str, status: str):
    """Автосохранение оценки: вызывается при изменении любого виджета
    (кнопок «сохранить» в интерфейсе нет — замечание Сергея 31.08)."""
    if not ss.check_id:
        return
    con = connect()
    try:
        upsert_feedback(
            con, ss.check_id, aspect_id, aspect_name, status,
            rating=rating_key(ss.get(f"rate_{aspect_id}", RATINGS[0])),
            note=ss.get(f"note_{aspect_id}", ""),
            note_type=ss.get(f"dec_{aspect_id}", "none"))
    finally:
        con.close()


def rating_label(rating: str | None) -> str:
    """Ключ базы ('up'/'down'/None) → подпись переключателя оценки."""
    return (RATINGS[1] if rating == "up"
            else RATINGS[2] if rating == "down" else RATINGS[0])


def rating_key(label) -> str | None:
    """Подпись переключателя → ключ базы."""
    s = str(label)
    return "up" if s.startswith("👍") else "down" if s.startswith("👎") else None


def restore_decisions(report: dict):
    """Поднять решения человека из базы в состояние виджетов (R-04).

    Streamlit удаляет из session_state значения виджетов, которые не
    отрисовались в текущем прогоне: ушёл на шаг 1 — ключи dec_/rate_/note_
    стёрты, вернулся на шаг 2 — виджеты показывают значения по умолчанию,
    а шаг 3 строит план так, будто решений не было. Автосохранение при этом
    писало всё в базу. Правило: перед отрисовкой шага 2 или 3 всё, чего нет
    в состоянии, берётся из базы; чего нет и в базе — остаётся по
    умолчанию. Ключ ставится ДО создания виджета — тогда Streamlit берёт
    его как текущее значение без предупреждений."""
    if not ss.check_id or not report:
        return
    con = connect()
    try:
        saved = decisions_for_check(con, ss.check_id)
    finally:
        con.close()
    ids = ([v["id"] for v in report.get("verdicts", [])]
           + [b["id"] for b in report.get("other_remarks", [])])
    for aid in ids:
        d = saved.get(aid)
        if not d:
            continue
        ss.setdefault(f"dec_{aid}", d["note_type"] if d["note_type"] in DECISIONS
                      else "none")
        ss.setdefault(f"rate_{aid}", rating_label(d["rating"]))
        ss.setdefault(f"note_{aid}", d["note"])


def decision_widgets(aspect_id: int, aspect_name: str, status: str,
                     default_decision: str, note_hint: str = ""):
    """Решение (слева, главное) и оценка ответа системы (справа) с
    автосохранением. Порядок по замечанию Сергея: сначала «что делать».

    Значение по умолчанию кладётся в session_state до создания виджета (а не
    через index=/value=), чтобы восстановленные из базы значения (R-04) и
    значения по умолчанию шли одним путём."""
    st.markdown("---")
    left, right = st.columns([2, 1])
    keys = list(DECISIONS)
    ss.setdefault(f"dec_{aspect_id}", default_decision)
    ss.setdefault(f"rate_{aspect_id}", RATINGS[0])
    ss.setdefault(f"note_{aspect_id}", "")
    left.selectbox(
        "Что делать с замечанием", keys,
        key=f"dec_{aspect_id}",
        format_func=lambda k: f"{DECISIONS[k][0]} — {DECISIONS[k][1]}",
        on_change=save_decision, args=(aspect_id, aspect_name, status))
    right.radio(
        "Оцените ответ системы", RATINGS, key=f"rate_{aspect_id}",
        help="Оценка не меняет вердикт: это журнал качества системы. "
             "«Система ошиблась» убирает пункт из плана работ.",
        on_change=save_decision, args=(aspect_id, aspect_name, status))
    st.text_input(
        "Своими словами (попадёт в план работ вместо формулировки системы)",
        key=f"note_{aspect_id}", placeholder=note_hint,
        on_change=save_decision, args=(aspect_id, aspect_name, status))


top_anchor()
st.title("🏷️ LabelCheck — проверка макета упаковки по ТР ЕАЭС")
st.caption("Инструмент предварительной проверки. Финальное решение — "
           "за специалистом и юристом.")
st.segmented_control("Шаг", STEPS, key="nav", label_visibility="collapsed")
step = ss.nav or STEPS[0]
if ss.pop("scroll_top", False):
    scroll_to_top()
st.divider()


# ═════════════════════════════ 1 · МАКЕТ ═════════════════════════════════════

if step == STEPS[0]:
    st.subheader("Шаг 1. Загрузите макет и проверьте распознанный текст")
    st.markdown("Система читает макет и раскладывает его на блоки. "
                "**Прочитанный текст можно поправить** — проверка пойдёт "
                "по исправленному тексту.")

    src = st.radio("Источник", ["Загрузить PDF-макет", "Открыть распознанный ранее"],
                   horizontal=True, label_visibility="collapsed")

    if src == "Загрузить PDF-макет":
        up = st.file_uploader("PDF-макет упаковки", type=["pdf"])
        if up is not None:
            UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            pdf_path = UPLOADS_DIR / up.name
            pdf_path.write_bytes(up.getvalue())
            layout_path = LAYOUTS_DIR / (pdf_path.stem + ".json")
            if layout_path.exists() and ss.layout_path != str(layout_path):
                st.info("Этот макет уже распознан — открыт готовый результат.")
                ss.layout = json.loads(layout_path.read_text(encoding="utf-8"))
                ss.layout_path, ss.report, ss.plan = str(layout_path), None, None
            if st.button("🔍 Распознать макет" if not layout_path.exists()
                         else "🔄 Распознать заново (≈ $0,10)", type="primary"):
                from labelcheck import vision
                with st.spinner("Читаю макет — 1–3 минуты…"):
                    layout = vision.analyze(pdf_path, out_path=layout_path,
                                            quiet=True)
                ss.layout, ss.layout_path = layout, str(layout_path)
                ss.report = ss.plan = ss.sel_region = None
                st.rerun()
    else:
        files = sorted(p for p in LAYOUTS_DIR.glob("*.json")
                       if not p.name.endswith(".orig.json"))
        if not files:
            st.info("Распознанных макетов пока нет — загрузите PDF.")
        else:
            def _layout_label(path: Path) -> str:
                # Двух layout'ов одного PDF в списке не отличить по имени файла
                # (R-40: канонический gyoza.json и копия по имени PDF) —
                # показываем исходный PDF и дату файла.
                try:
                    meta = json.loads(path.read_text(encoding="utf-8")).get("meta", {})
                except (OSError, json.JSONDecodeError):
                    meta = {}
                src = meta.get("source_pdf") or "?"
                when = time.strftime("%d.%m %H:%M", time.localtime(path.stat().st_mtime))
                return f"{path.stem}  ·  PDF: {src}  ·  файл от {when}"

            pick = st.selectbox("Макет", files, format_func=_layout_label)
            if st.button("📂 Открыть", type="primary"):
                ss.layout = json.loads(pick.read_text(encoding="utf-8"))
                ss.layout_path = str(pick)
                ss.report = ss.plan = ss.sel_region = None
                st.rerun()

    layout = ss.layout
    if layout:
        regions = layout.get("regions", [])
        st.divider()
        meta = layout.get("meta", {})
        cov = layout.get("text_layer_coverage")
        st.markdown(f"**{meta.get('source_pdf', '—')}** · блоков: {len(regions)}")
        if isinstance(cov, (int, float)):
            st.caption(f"Текст сверен с текстовым слоем PDF на {cov * 100:.0f}% — "
                       "опечатки и подмены букв отслеживаются автоматически.")
            unread = layout.get("unread_layer_words") or []
            if cov < CFG["vision"].get("coverage_warning_threshold", 0.9) and unread:
                with st.expander(f"⚠️ Не прочитано {len(unread)} слов текстового "
                                 f"слоя (покрытие {cov * 100:.0f}%) — возможно, "
                                 "пропущен или обрезан блок", expanded=False):
                    st.caption("Найдите эти слова на макете: если они в одном "
                               "месте — блок не прочитан целиком, распознайте "
                               "заново или допишите текст блока вручную.")
                    st.markdown(", ".join(unread))
            if layout.get("text_layer_partial"):
                share = layout.get("text_layer_invented_share") or 0
                st.warning(f"Часть текста этого макета — в кривых (≈{share * 100:.0f}% "
                           "прочитанных слов нет в текстовом слое). Для таких блоков "
                           "автоматическая проверка «слов вне слоя» отключена — "
                           "просмотрите их тексты внимательнее (R-23).")
        else:
            st.warning("В этом PDF нет текстового слоя (шрифты в кривых): "
                       "автоматическая сверка опечаток невозможна — "
                       "просмотрите тексты блоков внимательно.")
        if layout.get("missing"):
            st.warning("Не найдены обязательные блоки: " +
                       ", ".join(layout["missing"]))

        ids = [r["id"] for r in regions]
        if ss.sel_region not in ids:
            ss.sel_region = ids[0] if ids else None
        if "region_pick" not in ss and ss.sel_region:
            ss.region_pick = ss.sel_region

        pdf = find_source_pdf(layout)
        ids_list = ids
        sel_reg = next((r for r in regions if r["id"] == ss.sel_region), None)

        # ── СВЕРХУ: крупный вид блока (слева) и его текст (справа) ─────────
        # Порядок по замечанию Сергея (01.09): то, с чем работают руками —
        # выше; общий макет для навигации — ниже.
        top_left, top_right = st.columns([3, 2], gap="large")

        with top_left:
            if sel_reg and pdf:
                st.markdown(f"**Блок {sel_reg['id']} крупно** "
                            f"({sel_reg['kind']})")
                with st.container(horizontal=True):
                    idx = ids_list.index(sel_reg["id"])
                    if st.button("⬅️ предыдущий", key="prev_reg",
                                 disabled=idx == 0):
                        ss.sel_region = ids_list[idx - 1]
                        ss.region_pick = ss.sel_region
                        st.rerun()
                    if st.button("следующий ➡️", key="next_reg",
                                 disabled=idx == len(ids_list) - 1):
                        ss.sel_region = ids_list[idx + 1]
                        ss.region_pick = ss.sel_region
                        st.rerun()
                    st.segmented_control("Масштаб", ZOOM_STEPS, key="zoom",
                                         format_func=lambda z: f"{z}%",
                                         label_visibility="collapsed")
                zoom = ss.zoom or 100
                crop = region_crop(str(pdf), pdf.stat().st_mtime,
                                   json.dumps(sel_reg.get("bbox")),
                                   int(900 * zoom / 100))
                if crop is not None:
                    with st.container(height=520 if zoom > 100 else "content"):
                        st.image(crop, width=crop.width)
                    st.caption(f"Блок {idx + 1} из {len(ids_list)}. "
                               "Масштаб меняет размер этого фрагмента; при "
                               "увеличении окно прокручивается.")
                else:
                    st.caption("У блока нет рамки — крупный вид недоступен.")
            elif not pdf:
                st.info("PDF макета не найден рядом — тексты блоков доступны "
                        "справа.")

        with top_right:
            st.markdown("### Текст блока")
            manual = [r for r in regions if r.get("status") != "прочитано"]
            if manual:
                st.warning(f"Проверьте текст {len(manual)} блоков: "
                           + ", ".join(r["id"] for r in manual))

            def _pick_changed():
                ss.sel_region = ss.region_pick

            st.selectbox(
                "Блок", ids_list, key="region_pick", on_change=_pick_changed,
                format_func=lambda rid: (
                    ("✏️ " if next(r for r in regions if r["id"] == rid).get("human_edited")
                     else "✅ " if next(r for r in regions if r["id"] == rid).get("status") == "прочитано"
                     else "⚠️ ") + rid + " · " +
                    next(r for r in regions if r["id"] == rid)["kind"]))
            sel = ss.sel_region
            region = next(r for r in regions if r["id"] == sel)
            if region.get("status_reason"):
                st.caption(f"{region.get('status')} — {region['status_reason']}")
            elif region.get("layer_partial") and region.get("invented_words"):
                st.caption("прочитано · блок частично в кривых, в текстовом слое "
                           "нет слов: " + ", ".join(region["invented_words"][:8]))
            new_text = st.text_area(
                "Исправьте ошибки распознавания",
                value=region.get("text") or "", height=320, key=f"txt_{sel}")
            changed = new_text != (region.get("text") or "")
            unsure = region.get("status") != "прочитано"
            b_save, b_ok = st.columns(2)
            if b_save.button("💾 Сохранить исправление",
                             type="primary" if changed else "secondary",
                             disabled=not changed, width="stretch"):
                apply_region_edit(layout, sel, new_text)
                save_layout(layout, ss.layout_path)
                ss.report = ss.plan = None
                st.success("Исправление сохранено — проверка пойдёт по нему.")
                time.sleep(0.7)
                st.rerun()
            # Подтверждение без правки (R-08): сторожа слоя ошибаются, и у
            # человека должен быть способ снять пометку «требует проверки»,
            # не выдумывая правку текста.
            if b_ok.button("✅ Текст верный — подтвердить",
                           type="primary" if (unsure and not changed) else "secondary",
                           disabled=not unsure or changed, width="stretch",
                           help="Блок помечен сомнительным, но текст верный: "
                                "снять пометку. Активно только для блоков "
                                "«требует ручной проверки» и без несохранённых "
                                "правок."):
                confirm_region(layout, sel)
                save_layout(layout, ss.layout_path)
                st.success("Блок подтверждён.")
                time.sleep(0.5)
                st.rerun()
            if not changed:
                st.caption("Текст не менялся. " +
                           ("Если он верный — подтвердите блок; " if unsure else
                            "Если правки не нужны — ") +
                           "к проверке — кнопкой внизу.")
            layer_diff_table(region)

        # ── НИЖЕ: общий вид макета с подсветкой блоков ─────────────────────
        if pdf:
            st.markdown("**Весь макет** — кликните по блоку, чтобы открыть "
                        "его крупно и поправить текст")
            slim = json.dumps(
                [{"id": r["id"], "kind": r["kind"], "bbox": r.get("bbox"),
                  "status": r.get("status")} for r in regions],
                ensure_ascii=False)
            shown = overlay_image(str(pdf), pdf.stat().st_mtime, slim,
                                  ss.sel_region, 900)
            clicked, click_error = None, None
            image_coords = load_click_component()
            if image_coords is None:
                click_error = ss.get("click_component_error")
            else:
                try:
                    clicked = image_coords(shown, width="stretch",
                                           key="layout_click")
                except Exception as e:  # noqa: BLE001 — версия API могла
                    click_error = f"{type(e).__name__}: {e}"  # измениться
            if click_error:
                st.image(shown, width="stretch")
                with st.expander("⚠️ Выбор блока мышью недоступен — "
                                 "подробности", expanded=False):
                    st.markdown(
                        "Компонент кликов не запустился. Блоки переключаются "
                        "кнопками «предыдущий/следующий» и списком — работать "
                        "можно.\n\n"
                        f"**Причина:** `{click_error}`\n\n"
                        f"**Python приложения:** `{sys.executable}`\n\n"
                        "Если пакет установлен, но в причине ошибка импорта — "
                        "установка попала в другую среду. Проверьте в "
                        "терминале со включённым `.venv`:\n\n"
                        "```\npython -c \"import streamlit_image_coordinates "
                        "as m; print(m.__file__)\"\n```")
            elif clicked:
                # Компонент хранит ПОСЛЕДНИЙ клик и отдаёт его при каждой
                # перерисовке. Без памяти об обработанном клике старый клик
                # переигрывал кнопки «предыдущий/следующий» и откатывал выбор
                # на блок, по которому кликали раньше (R-03). Реагируем
                # только на клик, которого ещё не видели. Подпись клика —
                # его время (unix_time компонента): подпись по координатам
                # и размерам ломалась, когда после «Сохранить» страница
                # перестраивалась и картинка меняла ширину на пиксели —
                # старый клик выглядел новым (R-29).
                click_sig = clicked.get("unix_time") or (
                    round(clicked["x"] / (clicked.get("width") or 1), 4),
                    round(clicked["y"] / (clicked.get("height") or 1), 4))
                if click_sig != ss.get("last_click"):
                    ss.last_click = click_sig
                    # Компонент возвращает фактические размеры отображения —
                    # считаем по ним, иначе координаты «уезжают».
                    cw = clicked.get("width") or shown.width
                    ch = clicked.get("height") or shown.height
                    rid = region_at(regions, clicked["x"] / cw, clicked["y"] / ch)
                    if rid and rid != ss.sel_region:
                        ss.sel_region = rid
                        ss.region_pick = rid
                        st.rerun()
            st.caption("🟩 прочитано уверенно · 🟨 требует проверки человеком "
                       "· 🟦 выбранный блок")

        st.divider()
        st.button("➡️ Перейти к шагу 2: проверка по регламентам",
                  type="primary", on_click=go_to, args=(STEPS[1],))


# ═════════════════════════════ 2 · ПРОВЕРКА ══════════════════════════════════

elif step == STEPS[1]:
    st.subheader("Шаг 2. Проверьте макет по регламентам")
    if not ss.layout:
        st.info("Сначала откройте макет на шаге 1.")
        st.button("⬅️ К шагу 1", on_click=go_to, args=(STEPS[0],))
    else:
        # Профильные регламенты выбирает только человек (решение Сергея
        # 02.09, R-05): автоопределение по словам-маркерам убрано из
        # интерфейса — по упаковке вид продукции однозначно не определяется
        # («подавайте с мясом» в рекомендациях включало ТР на мясо). По
        # умолчанию — только базовые регламенты, переключатели видны всегда.
        with st.container(border=True):
            st.markdown("**Базовые регламенты — применяются всегда, "
                        "независимо от продукта:**")
            st.markdown(
                "- **ТР ТС 022/2011** — маркировка пищевой продукции "
                "(основной: наименование, состав, сроки, пищевая ценность, "
                "язык, шрифт)\n"
                "- **ТР ТС 021/2011** — безопасность пищевой продукции "
                "(условия хранения, знак обращения ЕАС)\n"
                "- **ТР ТС 029/2012** — пищевые добавки и ароматизаторы "
                "(классы добавок, Е-коды, предупредительные надписи)\n"
                "- **ТР ТС 005/2011** — упаковка (знаки материала, петля "
                "Мёбиуса, «бокал-вилка»)")
        with st.container(border=True):
            st.markdown("**Профильные регламенты — подключите, если продукт "
                        "к ним относится:**")
            picked = []
            with st.container(horizontal=True):
                for key, label in CATEGORY_LABELS.items():
                    if st.toggle(label, key=f"cat_{key}",
                                 help=CATEGORY_HINTS[key]):
                        picked.append(key)
            override = set(picked)
            if picked:
                st.caption("Подключены: " + "; ".join(
                    CATEGORY_HINTS[k] for k in picked) + ". У них свои "
                    "требования к наименованию, составу и хранению.")
            else:
                st.caption("Ничего не отмечено — проверка только по базовым "
                           "регламентам. Так и нужно для овощей, фруктов, ягод, "
                           "бакалеи: профильных регламентов для них нет.")

        unread = [r["id"] for r in ss.layout.get("regions", [])
                  if r.get("status") != "прочитано"]
        if unread:
            st.warning("Блоки без выверки: " + ", ".join(unread) +
                       ". Их текст попадёт в проверку с пометкой «прочитан "
                       "ненадёжно» — вернитесь на шаг 1, если хотите поправить.")

        st.caption("Первая проверка макета — примерно $1 и несколько минут. "
                   "Повторная проверка того же макета без исправлений "
                   "мгновенная и бесплатная. Как только текст блока изменён, "
                   "ответы пересчитываются заново.")

        if st.button("🚀 Проверить макет", type="primary"):
            status = st.status("Готовлю проверку…", expanded=True)
            with status:
                st.write("Загружаю базу регламентов (первый запуск ~3 минуты)…")
                retriever, client = get_retriever_and_client()
                bar = st.progress(0.0)
                label = st.empty()

                def progress(done, total, name):
                    bar.progress(min(1.0, done / total))
                    label.write(f"Проверка {min(done + 1, total)} из {total}: "
                                f"**{name}**")

                report = check_layout(ss.layout, retriever, client, CFG,
                                      categories_override=override,
                                      use_cache=True, progress_cb=progress)
                st.write("Составляю план работ…")
                plan = build_plan(report, client=client, cfg=CFG)
            status.update(label="Проверка завершена", state="complete",
                          expanded=False)

            UI_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            stem = Path(ss.layout_path).stem
            rpath = UI_REPORTS_DIR / f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}"
            rpath.with_suffix(".json").write_text(
                json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
            rpath.with_suffix(".md").write_text(render_markdown(report),
                                                encoding="utf-8")
            con = connect()
            ss.check_id = record_check(con, report, str(rpath.with_suffix(".md")))
            con.close()
            ss.report, ss.plan = report, plan
            for k in [k for k in ss if str(k).startswith(("dec_", "rate_", "note_"))]:
                del ss[k]          # оценки прошлой проверки не переносим
            st.rerun()

    report = ss.report
    if report:
        restore_decisions(report)   # R-04: решения из базы — до виджетов
        st.divider()
        counts = {s: sum(1 for v in report["verdicts"]
                         if v["status"] == s and v["applicable"])
                  for s in (STATUS_VIOLATION, STATUS_MANUAL, STATUS_COMPLIANT)}
        n_na = sum(1 for v in report["verdicts"] if not v["applicable"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔴 Возможные нарушения", counts[STATUS_VIOLATION])
        c2.metric("🟡 Нужна ручная проверка", counts[STATUS_MANUAL])
        c3.metric("🟢 Замечаний нет", counts[STATUS_COMPLIANT])
        c4.metric("⚪ Не относится к продукту", n_na)
        st.caption(human_summary(report["meta"]))
        st.markdown("Разверните замечание, выберите **что с ним делать** и "
                    "при желании оцените ответ системы. Всё сохраняется само.")

        order = {STATUS_VIOLATION: 0, STATUS_MANUAL: 1, STATUS_COMPLIANT: 2}
        for v in sorted(report["verdicts"],
                        key=lambda v: (order[v["status"]], v["id"])):
            icon = "⚪" if not v["applicable"] else ICONS[v["status"]]
            head = ("не относится к этому продукту" if not v["applicable"]
                    else v["status"])
            with st.expander(f"{icon} {v['id']}. {v['name']} — {head}",
                             expanded=(v["applicable"] and
                                       v["status"] == STATUS_VIOLATION)):
                st.write(v["explanation"] or "—")
                if v.get("arithmetic"):
                    a = v["arithmetic"]
                    st.info(f"Расчёт калорийности по белкам, жирам и углеводам: "
                            f"{a['calc_kcal']} ккал / {a['calc_kj']} кДж; "
                            f"на макете {a.get('stated_kcal', '—')} ккал / "
                            f"{a.get('stated_kj', '—')} кДж "
                            f"(расхождение {a.get('dev_kcal_pct', '—')}%).")
                default = ("designer" if v["status"] == STATUS_VIOLATION
                           else "manual" if v["status"] == STATUS_MANUAL
                           else "none")
                if not v["applicable"]:
                    default = "none"
                # Решение и оценка идут СРАЗУ после вердикта (замечание
                # Сергея 01.09): вердикт и поле «своими словами» должны быть
                # видны на экране вместе, без прокрутки через цитаты.
                decision_widgets(v["id"], v["name"], v["status"], default,
                                 note_hint="например: заменить «Изготовлено и "
                                           "упаковано» на «Дата изготовления»")
                if v["citations"] or v.get("downgraded_reason"):
                    with st.expander("📖 Нормы регламентов, на которые "
                                     "опирается вердикт", expanded=False):
                        for c in v["citations"]:
                            quote_block(c["quote"], c["address"])
                        if v.get("downgraded_reason"):
                            st.caption(f"⚙️ Система понизила уверенность: "
                                       f"{v['downgraded_reason']}")

        st.subheader("Прочие замечания (не требования регламентов)")
        for block in report["other_remarks"]:
            with st.expander(f"{block['id']}. {block['name']} "
                             f"({len(block['items'])})"):
                for item in block["items"]:
                    st.markdown(f"- {item}")
                decision_widgets(block["id"], block["name"], "прочее замечание",
                                 "none",
                                 note_hint="например: исправить опечатку "
                                           "«plaease» на «please»")

        with st.expander("Как система прочитала макет"):
            vis = report["vision"]
            cov = vis["text_layer_coverage"]
            st.markdown("- Сверка с текстовым слоем PDF: " +
                        (f"{cov * 100:.0f}% текста" if isinstance(cov, (int, float))
                         else "текстового слоя нет, сверка невозможна"))
            st.markdown("- Не найденные обязательные блоки: " +
                        (", ".join(vis["missing"]) or "нет"))
            for r in vis["manual_regions"]:
                st.markdown(f"- Блок {r['id']} ({r['kind']}) прочитан "
                            f"ненадёжно: {r['reason'] or '—'}")

        st.divider()
        st.button("➡️ Готово — перейти к плану работ", type="primary",
                  on_click=go_to, args=(STEPS[2],))


# ═════════════════════════════ 3 · ПЛАН РАБОТ ════════════════════════════════

elif step == STEPS[2]:
    st.subheader("Шаг 3. План работ")
    if not ss.report:
        st.info("Сначала проверьте макет на шаге 2.")
        st.button("⬅️ К шагу 2", on_click=go_to, args=(STEPS[1],))
    else:
        st.markdown("Короткие пункты без ссылок на регламенты — можно "
                    "скопировать в письмо дизайнеру или поставщику. "
                    "Учтены ваши решения и заметки с шага 2.")

        restore_decisions(ss.report)   # R-04: после захода на шаг 1 ключи стёрты
        decisions = {}
        for v in ss.report["verdicts"]:
            decisions[v["id"]] = {
                "rating": rating_key(ss.get(f"rate_{v['id']}", RATINGS[0])),
                "target": ss.get(f"dec_{v['id']}"),
                "note": ss.get(f"note_{v['id']}", "")}
        plan = apply_human_decisions(ss.plan or [], decisions)

        # Прочие замечания (штрихкод, орфография) — попадают в план только
        # если человек выбрал для них адресата (по умолчанию «не требуется»).
        # Каждая находка — отдельным пунктом: раньше в план шли первые две
        # из N и остальные пропадали (R-06). Своя формулировка человека
        # заменяет весь список одним пунктом.
        for block in ss.report["other_remarks"]:
            target = ss.get(f"dec_{block['id']}", "none")
            if target == "none" or ss.get(f"rate_{block['id']}") == RATINGS[2]:
                continue
            note = (ss.get(f"note_{block['id']}", "") or "").strip()
            texts = [note] if note else list(block["items"])
            for text in texts:
                plan.append({"aspect_id": block["id"],
                             "aspect_name": block["name"], "target": target,
                             "text": text, "edited_by_human": bool(note)})
        plan.sort(key=lambda i: (list(TARGETS).index(i["target"]), i["aspect_id"]))

        if not plan:
            st.success("Действий не требуется — замечаний, требующих работы, "
                       "не осталось.")
        for key, (title, hint) in TARGETS.items():
            items = [i for i in plan if i["target"] == key]
            if not items:
                continue
            st.markdown(f"### {title}")
            st.caption(hint)
            for n, item in enumerate(items, 1):
                mark = (" <em>(ваша формулировка)</em>"
                        if item.get("edited_by_human") else "")
                st.markdown(f"{n}. {item['text']}  \n"
                            f"<span style='opacity:.6;font-size:.85em'>"
                            f"проверка: {item['aspect_name']}{mark}</span>",
                            unsafe_allow_html=True)

        md = render_plan_markdown(plan, ss.report)
        st.markdown("#### Скопировать или скачать")
        # Кнопка копирования — отдельной строкой: она рисуется во встроенной
        # рамке браузера и в одном ряду с обычными кнопками съезжает вниз
        # (замечание Сергея 01.09).
        copy_button(md)
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Скачать текстом (.md)", data=md,
                           file_name="Позиции_к_доработке.md",
                           mime="text/markdown", width="stretch")
        docx_path = UI_REPORTS_DIR / "plan_last.docx"
        try:
            UI_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            plan_to_docx(plan, ss.report, docx_path)
            d2.download_button(
                "⬇️ Скачать для Word (.docx)", data=docx_path.read_bytes(),
                file_name="Позиции_к_доработке.docx", width="stretch",
                mime="application/vnd.openxmlformats-officedocument."
                     "wordprocessingml.document")
        except Exception as e:  # noqa: BLE001
            d2.caption(f"Word-версия недоступна: {e}")
        st.markdown("#### Текст плана целиком")
        # Показываем ЦЕЛИКОМ, без внутренней прокрутки (решение Сергея):
        # текст должен читаться и копироваться одним куском, какой бы
        # длинный он ни был. За возврат страницы наверх при смене шага
        # отвечает якорь верха (scroll_to_top), а не высота этого блока.
        if True:
            # st.code не переносит длинные строки — текст уезжал за край окна.
            st.markdown(
                "<div style='white-space:pre-wrap;word-break:break-word;"
                "font-family:ui-monospace,SFMono-Regular,Menlo,monospace;"
                "font-size:.9em;background:rgba(150,150,150,.08);"
                "padding:12px;border-radius:.5rem;max-width:100%;'>"
                + html.escape(md) + "</div>", unsafe_allow_html=True)

    st.divider()
    with st.expander("Журнал проверок (история и оценки)"):
        con = connect()
        checks = fetch_checks(con)
        if not checks:
            st.caption("Проверок ещё не было.")
        else:
            for c in checks[:10]:
                st.markdown(f"**#{c['id']}** · {c['ts']} · "
                            f"{c['source_pdf'] or '—'} — "
                            f"🔴 {c['n_violation']} · 🟡 {c['n_manual']} · "
                            f"🟢 {c['n_ok']} · ⚪ {c['n_na']}")
            fb = fetch_feedback(con)
            if fb:
                st.markdown(f"Оценок сохранено: **{len(fb)}** "
                            f"(👍 {sum(1 for f in fb if f['rating'] == 'up')} · "
                            f"👎 {sum(1 for f in fb if f['rating'] == 'down')})")
                st.download_button(
                    "⬇️ Выгрузить все оценки (JSON)",
                    data=json.dumps(fb, ensure_ascii=False, indent=1),
                    file_name="labelcheck_feedback.json", mime="application/json")
        con.close()


# ═════════════════════════════ 4 · МОНИТОРИНГ ════════════════════════════════

else:
    # Имя MAIN-модели из .env нужно, чтобы отличить полный прогон от повтора
    # из кэша (по числу вызовов); без .env берётся модель с наибольшим
    # расходом токенов.
    import os
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    dashboard_ui.render(CFG, LAYOUTS_DIR, os.environ.get("MAIN_MODEL"))
