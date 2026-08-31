"""Streamlit UI LabelCheck (блок C, день 9). Запуск с мака:

    streamlit run labelcheck/app.py

Три шага по сценарию ТЗ §2, каждый заканчивается кнопкой перехода:
1. «Макет» — PDF → vision → просмотр с зумом и кликом по регионам, правка
   текста ЧЕЛОВЕКОМ до вердиктов (правки в layout-JSON, история в edits).
2. «Проверка» — выбор профильных регламентов → вердикты по 21 аспекту →
   отчёт с цитатами; под каждым вердиктом оценка ответа системы и решение,
   что с ним делать (всё пишется в SQLite, labelcheck/store.py).
3. «План работ» — короткие списки без цитат регламентов: дизайнеру,
   поставщику, проверить самому; выгрузка в Markdown и Word.

Правки UI по замечаниям Сергея (31.08): человеческие формулировки вместо
внутренних терминов, категории кнопками, живой прогресс по аспектам,
фидбек рядом с вердиктом, план работ как итоговый документ.
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
from labelcheck.retrieval import ROOT, Retriever, load_config
from labelcheck.store import (apply_region_edit, connect, fetch_checks,
                              fetch_feedback, record_check, record_feedback,
                              save_layout)
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

# Профильные регламенты человеческим языком (внутренние ключи не показываем).
CATEGORY_LABELS = {
    "meat": "🥩 Мясная продукция",
    "poultry": "🍗 Продукция из мяса птицы",
    "fish": "🐟 Рыба и морепродукты",
}
CATEGORY_HINTS = {
    "meat": "ТР ТС 034/2013 — мясо и мясная продукция",
    "poultry": "ТР ЕАЭС 051/2021 — мясо птицы",
    "fish": "ТР ЕАЭС 040/2016 — рыба и морепродукты",
}
# Что делать с замечанием: ключ → (подпись, пояснение)
DECISIONS = {
    "none": ("Ничего не требуется", "замечание снято"),
    "designer": ("Замечание дизайнеру", "поправить в макете"),
    "supplier": ("Запросить у поставщика", "нужны данные производителя"),
    "manual": ("Проверить самому", "смотрю вручную"),
}


# ── кэшируемые тяжёлые объекты ───────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_retriever_and_client():
    """Индексы и OpenAI-клиент — один раз на процесс Streamlit."""
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(ROOT / ".env")
    client = OpenAI()
    retriever = Retriever(CFG, openai_client=client)
    return retriever, client


@st.cache_data(show_spinner=False)
def rendered_page(pdf_path: str, mtime: float):
    from labelcheck import render
    return render.render_page(pdf_path, scale=2)


def draw_regions(img, regions, selected_id=None, label_regions=True):
    """Рамки регионов с подписями (номер и тип) поверх рендера."""
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    w, h = out.size
    size = max(13, int(min(w, h) * 0.016))
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
        d.rectangle([x0, y0, x1, y1], outline=color, width=5 if chosen else 3)
        if label_regions:
            tag = f"{r['id']} · {r['kind']}"
            tw = d.textlength(tag, font=font)
            ty = max(0, y0 - size - 6)
            d.rectangle([x0, ty, x0 + tw + 8, ty + size + 6], fill=color)
            d.text((x0 + 4, ty + 3), tag, fill=(255, 255, 255), font=font)
    return out


def region_at(regions, x_rel, y_rel):
    """Регион под кликом (самый маленький из попавших — вложенные рамки)."""
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
    """Цитата пункта регламента. Через HTML, а не markdown: текст пункта
    часто начинается с «3. …» — markdown превращал его в нумерованный
    список и ломал отступы и цвет (замечание Сергея 31.08)."""
    body = html.escape(text).replace("\n", "<br>")
    st.markdown(
        f"<div style='border-left:3px solid #9aa0a6;padding:6px 12px;"
        f"margin:6px 0;background:rgba(150,150,150,.08);color:inherit;'>"
        f"<div style='opacity:.75;font-size:.85em;margin-bottom:4px'>"
        f"📖 {html.escape(address)}</div>{body}</div>",
        unsafe_allow_html=True)


def human_summary(meta: dict) -> str:
    """Подпись отчёта человеческим языком (было: «охват детекта: targeted»)."""
    cats = meta.get("categories") or {}
    if cats:
        names = ", ".join(CATEGORY_LABELS.get(c, c).split(" ", 1)[-1] for c in cats)
        scan = {"targeted": "определено автоматически по названию и составу",
                "full": "определено автоматически по всему макету",
                "manual": "указано вручную"}.get(meta.get("category_scan"), "")
        cat_txt = f"Профильные регламенты: {names} ({scan})"
    else:
        cat_txt = ("Профильные регламенты не применялись: мясо, птица и рыба "
                   "в продукте не обнаружены — проверка по общим правилам "
                   "маркировки")
    sec = meta.get("seconds") or 0
    dur = (f"{int(sec // 60)} мин {int(sec % 60)} с" if sec >= 60
           else f"{int(sec)} с")
    return f"{cat_txt}. Проверка заняла {dur}."


# ── состояние сессии ─────────────────────────────────────────────────────────

ss = st.session_state
for key, default in (("layout", None), ("layout_path", None), ("report", None),
                     ("check_id", None), ("plan", None), ("sel_region", None),
                     ("zoom", 1.0), ("step", 0), ("saved_ids", set())):
    ss.setdefault(key, default)

st.title("🏷️ LabelCheck — проверка макета упаковки по ТР ЕАЭС")
st.caption("Инструмент предварительной проверки. Финальное решение — "
           "за специалистом и юристом.")

tabs = st.tabs(["1 · Макет", "2 · Проверка", "3 · План работ"])


# ═════════════════════════════ 1 · МАКЕТ ═════════════════════════════════════

with tabs[0]:
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
            pick = st.selectbox("Макет", files, format_func=lambda p: p.stem)
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
            st.caption(f"Текст макета сверен с текстовым слоем PDF на "
                       f"{cov * 100:.0f}% — опечатки и подмены букв "
                       f"отслеживаются автоматически.")
        else:
            st.warning("В этом PDF нет текстового слоя (шрифты в кривых): "
                       "автоматическая сверка опечаток невозможна — "
                       "просмотрите тексты блоков внимательно.")
        if layout.get("missing"):
            st.warning("Не найдены обязательные блоки: " +
                       ", ".join(layout["missing"]))

        if ss.sel_region not in [r["id"] for r in regions]:
            ss.sel_region = regions[0]["id"] if regions else None

        col_img, col_edit = st.columns([3, 2], gap="large")

        with col_img:
            pdf = find_source_pdf(layout)
            zc1, zc2, zc3, zc4 = st.columns([1, 1, 1, 3])
            if zc1.button("➖", help="Уменьшить"):
                ss.zoom = max(0.5, round(ss.zoom - 0.25, 2))
            if zc2.button("➕", help="Увеличить"):
                ss.zoom = min(4.0, round(ss.zoom + 0.25, 2))
            if zc3.button("↺", help="Вернуть масштаб 100%"):
                ss.zoom = 1.0
            zc4.caption(f"Масштаб {int(ss.zoom * 100)}%. "
                        "Кликните по блоку на макете, чтобы открыть его текст.")
            if pdf:
                base = rendered_page(str(pdf), pdf.stat().st_mtime)
                img = draw_regions(base, regions, selected_id=ss.sel_region)
                view_w = int(900 * ss.zoom)
                shown = img.resize((view_w,
                                    max(1, int(img.height * view_w / img.width))),
                                   Image.LANCZOS)
                clicked = None
                try:
                    from streamlit_image_coordinates import \
                        streamlit_image_coordinates as image_coords
                    clicked = image_coords(shown, key="layout_click")
                except ImportError:
                    st.image(shown)
                    st.caption("Для выбора блока мышью установите пакет "
                               "streamlit-image-coordinates.")
                if clicked:
                    rid = region_at(regions, clicked["x"] / shown.width,
                                    clicked["y"] / shown.height)
                    if rid and rid != ss.sel_region:
                        ss.sel_region = rid
                        st.rerun()
                st.caption("🟩 прочитано уверенно · 🟨 требует проверки "
                           "человеком · 🟦 выбранный блок. "
                           "При масштабе больше 100% страница прокручивается.")
            else:
                st.info("PDF макета не найден рядом — тексты блоков доступны "
                        "справа.")

        with col_edit:
            st.markdown("### Блоки макета")
            manual = [r for r in regions if r.get("status") != "прочитано"]
            if manual:
                st.warning(f"Проверьте текст {len(manual)} блоков: "
                           + ", ".join(r["id"] for r in manual))
            ids = [r["id"] for r in regions]
            sel = st.selectbox(
                "Блок", ids, index=ids.index(ss.sel_region) if ss.sel_region in ids else 0,
                format_func=lambda rid: (
                    ("✏️ " if next(r for r in regions if r["id"] == rid).get("human_edited")
                     else "✅ " if next(r for r in regions if r["id"] == rid).get("status") == "прочитано"
                     else "⚠️ ") + rid + " · " +
                    next(r for r in regions if r["id"] == rid)["kind"]))
            if sel != ss.sel_region:
                ss.sel_region = sel
                st.rerun()
            region = next(r for r in regions if r["id"] == sel)
            if region.get("status_reason"):
                st.caption(f"{region.get('status')} — {region['status_reason']}")
            new_text = st.text_area(
                "Текст блока — исправьте ошибки распознавания",
                value=region.get("text") or "", height=280, key=f"txt_{sel}")
            changed = new_text != (region.get("text") or "")
            if st.button("💾 Сохранить исправление", type="primary" if changed else "secondary",
                         disabled=not changed,
                         help="Обычная кнопка: нажмите мышью, никаких "
                              "сочетаний клавиш не нужно"):
                apply_region_edit(layout, sel, new_text)
                save_layout(layout, ss.layout_path)
                ss.report = ss.plan = None
                st.success("Исправление сохранено — проверка пойдёт по нему.")
                time.sleep(0.8)
                st.rerun()
            if not changed:
                st.caption("Текст не менялся. Если правки не нужны — переходите "
                           "к проверке кнопкой ниже.")

        st.divider()
        st.markdown("#### Тексты проверены?")
        st.button("➡️ Перейти к шагу 2: проверка по регламентам",
                  type="primary", key="goto_check",
                  help="Откройте вкладку «2 · Проверка» сверху")
        st.caption("Нажмите вкладку «2 · Проверка» вверху страницы — "
                   "исправления уже сохранены.")


# ═════════════════════════════ 2 · ПРОВЕРКА ══════════════════════════════════

with tabs[1]:
    st.subheader("Шаг 2. Проверьте макет по регламентам")
    if not ss.layout:
        st.info("Сначала откройте макет на вкладке «1 · Макет».")
    else:
        st.markdown(
            "**Выберите профильные регламенты.** Общие правила маркировки "
            "(ТР ТС 022/2011 и др.) применяются всегда. Дополнительно "
            "существуют регламенты для отдельных видов продукции — у них "
            "свои требования к наименованию, составу и хранению.")
        mode = st.radio(
            "Как определить профильные регламенты",
            ["Определить автоматически по названию и составу",
             "Указать вручную",
             "Не применять профильные — только общие правила маркировки"],
            captions=[
                "Система ищет в тексте макета признаки мяса, птицы или рыбы",
                "Вы сами отмечаете, к какой продукции относится макет",
                "Подходит для овощей, фруктов, бакалеи — там профильных "
                "регламентов нет"],
            label_visibility="collapsed")

        override = None
        if mode == "Указать вручную":
            picked = []
            cols = st.columns(len(CATEGORY_LABELS))
            for col, (key, label) in zip(cols, CATEGORY_LABELS.items()):
                on = col.toggle(label, key=f"cat_{key}", help=CATEGORY_HINTS[key])
                if on:
                    picked.append(key)
            override = set(picked)
            if not picked:
                st.caption("Ни один переключатель не включён — проверка "
                           "пойдёт только по общим правилам.")
        elif mode.startswith("Не применять"):
            override = set()

        unread = [r["id"] for r in ss.layout.get("regions", [])
                  if r.get("status") != "прочитано"]
        if unread:
            st.warning("Блоки без выверки: " + ", ".join(unread) +
                       ". Их текст попадёт в проверку с пометкой "
                       "«прочитан ненадёжно» — вернитесь на шаг 1, если "
                       "хотите поправить.")

        st.caption("Первая проверка макета — примерно $1 и несколько минут. "
                   "Повторная проверка того же макета без исправлений "
                   "мгновенная и бесплатная: ответы берутся из сохранённых. "
                   "Как только вы исправите текст блока, ответы "
                   "пересчитываются заново.")

        if st.button("🚀 Проверить макет", type="primary"):
            status = st.status("Готовлю проверку…", expanded=True)
            with status:
                st.write("Загружаю базу регламентов (первый запуск ~3 минуты)…")
                retriever, client = get_retriever_and_client()
                bar = st.progress(0.0)
                label = st.empty()

                def progress(done, total, name):
                    bar.progress(done / total)
                    label.write(f"Проверка {done} из {total}: **{name}**")

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
            ss.report, ss.plan, ss.saved_ids = report, plan, set()
            st.rerun()

    report = ss.report
    if report:
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
        st.markdown("Разверните замечание, чтобы увидеть подробности, "
                    "нормы регламентов и **оценить ответ системы**.")

        con = connect()
        order = {STATUS_VIOLATION: 0, STATUS_MANUAL: 1, STATUS_COMPLIANT: 2}
        for v in sorted(report["verdicts"],
                        key=lambda v: (order[v["status"]], v["id"])):
            icon = "⚪" if not v["applicable"] else ICONS[v["status"]]
            head = ("не относится к этому продукту" if not v["applicable"]
                    else v["status"])
            saved = v["id"] in ss.saved_ids
            with st.expander(f"{icon} {v['id']}. {v['name']} — {head}"
                             + ("  ✔ оценено" if saved else ""),
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
                for c in v["citations"]:
                    quote_block(c["quote"], c["address"])
                if v.get("downgraded_reason"):
                    st.caption(f"⚙️ Система понизила уверенность: "
                               f"{v['downgraded_reason']}")

                st.markdown("---")
                fb1, fb2 = st.columns([1, 2])
                rating = fb1.radio(
                    "Оцените ответ системы",
                    ["не оценивал", "👍 верно", "👎 система ошиблась"],
                    key=f"rate_{v['id']}", horizontal=False,
                    help="Оценка не меняет вердикт — это журнал качества "
                         "системы. «Система ошиблась» убирает пункт из плана "
                         "работ и попадает в статистику для доработки правил.")
                if rating.startswith("👍") and v["status"] == STATUS_COMPLIANT:
                    default_dec = "none"
                elif rating.startswith("👎"):
                    default_dec = "none"
                elif v["status"] == STATUS_VIOLATION:
                    default_dec = "designer"
                elif v["status"] == STATUS_MANUAL:
                    default_dec = "manual"
                else:
                    default_dec = "none"
                dec_keys = list(DECISIONS)
                decision = fb2.selectbox(
                    "Что делать с замечанием",
                    dec_keys, index=dec_keys.index(default_dec),
                    key=f"dec_{v['id']}",
                    format_func=lambda k: f"{DECISIONS[k][0]} — {DECISIONS[k][1]}")
                note = st.text_input(
                    "Своими словами (попадёт в план работ вместо "
                    "формулировки системы)", key=f"note_{v['id']}",
                    placeholder="например: заменить «Изготовлено и упаковано» "
                                "на «Дата изготовления»")
                if st.button("💾 Сохранить оценку", key=f"save_{v['id']}"):
                    with st.spinner("Сохраняю…"):
                        record_feedback(
                            con, ss.check_id, v["id"], v["name"], v["status"],
                            rating=("up" if rating.startswith("👍")
                                    else "down" if rating.startswith("👎") else None),
                            note=note, note_type=DECISIONS[decision][0])
                        ss.saved_ids = ss.saved_ids | {v["id"]}
                    st.success("Сохранено")
        con.close()

        st.subheader("Прочие замечания (не требования регламентов)")
        for block in report["other_remarks"]:
            with st.expander(f"{block['id']}. {block['name']} "
                             f"({len(block['items'])})"):
                for item in block["items"]:
                    st.markdown(f"- {item}")

        vis = report["vision"]
        with st.expander("Как система прочитала макет"):
            cov = vis["text_layer_coverage"]
            st.markdown("- Сверка с текстовым слоем PDF: " +
                        (f"{cov * 100:.0f}% текста" if isinstance(cov, (int, float))
                         else "текстового слоя нет, сверка невозможна"))
            st.markdown("- Не найденные обязательные блоки: " +
                        (", ".join(vis["missing"]) or "нет"))
            for r in vis["manual_regions"]:
                st.markdown(f"- Блок {r['id']} ({r['kind']}) прочитан "
                            f"ненадёжно: {r['reason'] or '—'}")

        st.info("Оценили замечания? Откройте вкладку «3 · План работ» — "
                "там готовый документ для дизайнера и поставщика.")


# ═════════════════════════════ 3 · ПЛАН РАБОТ ════════════════════════════════

with tabs[2]:
    st.subheader("Шаг 3. План работ")
    if not ss.report:
        st.info("Сначала проверьте макет на вкладке «2 · Проверка».")
    else:
        st.markdown("Короткие пункты без ссылок на регламенты — можно "
                    "скопировать в письмо дизайнеру или поставщику. "
                    "Учтены ваши оценки и заметки с шага 2.")
        decisions = {}
        for v in ss.report["verdicts"]:
            rating = ss.get(f"rate_{v['id']}", "не оценивал")
            decisions[v["id"]] = {
                "rating": ("up" if str(rating).startswith("👍")
                           else "down" if str(rating).startswith("👎") else None),
                "target": ss.get(f"dec_{v['id']}"),
                "note": ss.get(f"note_{v['id']}", ""),
            }
        plan = apply_human_decisions(ss.plan or [], decisions)

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
                mark = " ✏️" if item.get("edited_by_human") else ""
                st.markdown(f"{n}. {item['text']}{mark}  \n"
                            f"<span style='opacity:.6;font-size:.85em'>"
                            f"проверка: {item['aspect_name']}</span>",
                            unsafe_allow_html=True)

        md = render_plan_markdown(plan, ss.report)
        st.markdown("#### Скопировать или скачать")
        st.code(md, language="markdown")
        st.caption("У блока выше справа есть кнопка копирования — "
                   "текст копируется целиком, без горячих клавиш.")
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Скачать текстом (.md)", data=md,
                           file_name="Позиции_к_доработке.md", mime="text/markdown")
        docx_path = UI_REPORTS_DIR / "plan_last.docx"
        try:
            UI_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            plan_to_docx(plan, ss.report, docx_path)
            d2.download_button(
                "⬇️ Скачать для Word (.docx)", data=docx_path.read_bytes(),
                file_name="Позиции_к_доработке.docx",
                mime="application/vnd.openxmlformats-officedocument."
                     "wordprocessingml.document")
        except Exception as e:  # noqa: BLE001
            d2.caption(f"Word-версия недоступна: {e}")

    st.divider()
    with st.expander("Журнал проверок (история и оценки)"):
        con = connect()
        checks = fetch_checks(con)
        if not checks:
            st.caption("Проверок ещё не было.")
        else:
            for c in checks[:10]:
                st.markdown(
                    f"**#{c['id']}** · {c['ts']} · {c['source_pdf'] or '—'} — "
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
