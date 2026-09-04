"""Шаг «4 · Мониторинг» (День 10): графики по журналу проверок.

Только отрисовка: все цифры готовит labelcheck/dashboard.py (без
Streamlit, с тестами). Графики — altair (стоит как зависимость Streamlit,
plotly не добавляем). Вызывается из app.py одной функцией render().

Семь графиков (рубрика: 5+):
1. статусы вердиктов по прогонам;
2. проблемные аспекты — доля «нарушение + ручная» среди применимых
   (последний прогон каждого макета);
3. согласие эксперта с системой — 👍/👎 по аспектам;
4. адресаты решений — дизайнеру / поставщику / проверить самому / снято;
5. стоимость прогонов (полные против повторов из кэша);
6. динамика: когда проверяли и сколько это занимало;
7. качество распознавания по макетам (из layout-JSON).
"""

import altair as alt
import pandas as pd
import streamlit as st

from labelcheck import dashboard as D

STATUS_COLORS = {D.STATUS_VIOLATION: "#e5484d", D.STATUS_MANUAL: "#e0a100",
                 D.STATUS_COMPLIANT: "#2ea043", "не относится": "#9aa0a6"}
RUN_COLORS = {"полный прогон": "#4c78a8", "повтор из кэша": "#b3b3b3"}
VISION_COLORS = {"прочитано без вмешательства": "#2ea043",
                 "подтверждено человеком": "#7cc27a",
                 "правлено человеком": "#4c78a8",
                 "ещё сомнительные": "#e0a100"}


def _chart(df: pd.DataFrame) -> alt.Chart:
    return alt.Chart(df)


# Высота в Streamlit — на весь виджет вместе с осями и легендой (autosize
# fit), поэтому к площади графика добавляется запас под подписи оси X и
# легенду внизу.
AXES_PX = 110


def _rows_height(n: int, per_row: int = 24) -> int:
    return max(220, per_row * n + AXES_PX)


def _short(name: str, n: int = 22) -> str:
    """Подпись оси: длинные имена файлов макетов режутся, чтобы Vega не
    прятал соседние подписи как перекрывающиеся."""
    return name if len(name) <= n else name[:n - 1] + "…"


def _run_label(c: dict) -> str:
    return f"#{c['id']} · {_short(c['layout'], 16)}"


def _legend(columns: int | None = None) -> alt.Legend:
    """Легенда внизу; columns — перенос на строки, когда график в узкой
    колонке и подписи не помещаются в одну."""
    kw = {"columns": columns} if columns else {}
    return alt.Legend(title=None, orient="bottom", labelLimit=300, **kw)


def _count_axis(title: str) -> alt.Axis:
    """Ось счётчика: целые деления, без «1.0, 2.0»."""
    return alt.Axis(title=title, format="d", tickMinStep=1)


def _y_names(order: list[str]) -> alt.Y:
    """Категориальная ось Y с полными подписями, без прореживания."""
    return alt.Y("аспект:N", sort=order, title=None,
                 axis=alt.Axis(labelLimit=260, labelOverlap=False))


def _x_runs(order: list[str]) -> alt.X:
    return alt.X("прогон:N", sort=order, title=None,
                 axis=alt.Axis(labelAngle=-45, labelLimit=160, labelOverlap=False))


def _status_by_check(checks: list[dict]):
    rows = []
    for c in checks:
        label = _run_label(c)
        for status, col in ((D.STATUS_VIOLATION, "n_violation"),
                            (D.STATUS_MANUAL, "n_manual"),
                            (D.STATUS_COMPLIANT, "n_ok"), ("не относится", "n_na")):
            rows.append({"прогон": label, "id": c["id"], "статус": status,
                         "аспектов": c.get(col) or 0})
    df = pd.DataFrame(rows)
    order = [_run_label(c) for c in checks]
    return (_chart(df).mark_bar().encode(
        x=_x_runs(order),
        y=alt.Y("аспектов:Q", axis=_count_axis("аспектов")),
        color=alt.Color("статус:N", scale=alt.Scale(domain=list(STATUS_COLORS),
                                                     range=list(STATUS_COLORS.values())),
                        legend=_legend()),
        order=alt.Order("статус:N"),
        tooltip=["прогон", "статус", "аспектов"])
        .properties(height=320))


def _problem_aspects(aspects: list[dict]):
    rows = []
    for a in aspects:
        if not a["applicable"]:
            continue
        name = f"{a['aspect_id']}. {a['aspect']}"
        for status, col in ((D.STATUS_VIOLATION, "violation"),
                            (D.STATUS_MANUAL, "manual"), (D.STATUS_COMPLIANT, "ok")):
            rows.append({"аспект": name, "статус": status, "прогонов": a[col],
                         "доля проблемных": a["problem_share"]})
    df = pd.DataFrame(rows)
    order = [f"{a['aspect_id']}. {a['aspect']}" for a in aspects if a["applicable"]]
    return (_chart(df).mark_bar().encode(
        y=_y_names(order),
        x=alt.X("прогонов:Q", stack="normalize", title="доля прогонов",
                axis=alt.Axis(format="%")),
        color=alt.Color("статус:N", scale=alt.Scale(domain=list(STATUS_COLORS)[:3],
                                                     range=list(STATUS_COLORS.values())[:3]),
                        legend=_legend(columns=2)),
        order=alt.Order("статус:N"),
        tooltip=["аспект", "статус", "прогонов",
                 alt.Tooltip("доля проблемных:Q", format=".0%")])
        .properties(height=_rows_height(len(order))))


def _ratings(ratings: list[dict]):
    rows = []
    for r in ratings:
        name = f"{r['aspect_id']}. {r['aspect']}"
        rows.append({"аспект": name, "оценка": "👍 верно", "оценок": r["up"],
                     "согласие": r["agreement"]})
        rows.append({"аспект": name, "оценка": "👎 система ошиблась", "оценок": r["down"],
                     "согласие": r["agreement"]})
    df = pd.DataFrame(rows)
    order = [f"{r['aspect_id']}. {r['aspect']}" for r in ratings]
    return (_chart(df).mark_bar().encode(
        y=_y_names(order),
        x=alt.X("оценок:Q", axis=_count_axis("оценок эксперта")),
        color=alt.Color("оценка:N", scale=alt.Scale(domain=["👍 верно", "👎 система ошиблась"],
                                                     range=["#2ea043", "#e5484d"]),
                        legend=_legend()),
        order=alt.Order("оценка:N"),
        tooltip=["аспект", "оценка", "оценок", alt.Tooltip("согласие:Q", format=".0%")])
        .properties(height=_rows_height(len(order))))


def _decisions(decisions: list[dict]):
    df = pd.DataFrame([{"решение": d["label"], "замечаний": d["count"]} for d in decisions])
    return (_chart(df).mark_bar(color="#4c78a8").encode(
        x=alt.X("решение:N", sort=[d["label"] for d in decisions], title=None,
                axis=alt.Axis(labelAngle=0, labelLimit=200)),
        y=alt.Y("замечаний:Q", axis=_count_axis("замечаний")),
        tooltip=["решение", "замечаний"])
        .properties(height=240))


def _costs(checks: list[dict]):
    rows = [{"прогон": _run_label(c), "id": c["id"],
             "стоимость, $": c["cost_usd"] if c["cost_usd"] is not None else 0.0,
             "токенов": c["tokens_total"], "вызовов MAIN": c["main_calls"],
             "тип": "полный прогон" if c["full_run"] else "повтор из кэша"}
            for c in checks]
    df = pd.DataFrame(rows)
    order = [r["прогон"] for r in rows]
    return (_chart(df).mark_bar().encode(
        x=_x_runs(order),
        y=alt.Y("стоимость, $:Q", title="$"),
        color=alt.Color("тип:N", scale=alt.Scale(domain=list(RUN_COLORS),
                                                  range=list(RUN_COLORS.values())),
                        legend=_legend()),
        tooltip=["прогон", "тип", alt.Tooltip("стоимость, $:Q", format=".2f"),
                 "токенов", "вызовов MAIN"])
        .properties(height=320))


def _timeline(checks: list[dict]):
    rows = [{"когда": c["when"], "минут": c["minutes"],
             "прогон": _run_label(c),
             "тип": "полный прогон" if c["full_run"] else "повтор из кэша"}
            for c in checks if c["when"] is not None]
    df = pd.DataFrame(rows)
    return (_chart(df).mark_circle(size=110).encode(
        x=alt.X("когда:T", title=None, axis=alt.Axis(format="%d.%m %H:%M")),
        y=alt.Y("минут:Q", title="длительность, мин"),
        color=alt.Color("тип:N", scale=alt.Scale(domain=list(RUN_COLORS),
                                                  range=list(RUN_COLORS.values())),
                        legend=_legend()),
        tooltip=["прогон", alt.Tooltip("когда:T", format="%d.%m.%Y %H:%M"), "минут", "тип"])
        .properties(height=240))


def _vision(vision: list[dict]):
    rows = []
    for v in vision:
        untouched = max(0, v["regions"] - v["manual"] - v["edited"] - v["confirmed"])
        for group, n in (("прочитано без вмешательства", untouched),
                         ("подтверждено человеком", v["confirmed"]),
                         ("правлено человеком", v["edited"]),
                         ("ещё сомнительные", v["manual"])):
            rows.append({"макет": _short(v["layout"], 28), "группа": group, "блоков": n})
    df = pd.DataFrame(rows)
    return (_chart(df).mark_bar().encode(
        y=alt.Y("макет:N", title=None, sort=[_short(v["layout"], 28) for v in vision],
                axis=alt.Axis(labelLimit=260, labelOverlap=False)),
        x=alt.X("блоков:Q", axis=_count_axis("блоков макета")),
        color=alt.Color("группа:N", scale=alt.Scale(domain=list(VISION_COLORS),
                                                     range=list(VISION_COLORS.values())),
                        legend=_legend()),
        order=alt.Order("группа:N"),
        tooltip=["макет", "группа", "блоков"])
        .properties(height=_rows_height(len(vision), 40)))


def _vision_table(vision: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "макет": v["layout"], "блоков": v["regions"],
        "покрытие слоя": (f"{v['coverage']:.0%}" if v.get("coverage") is not None
                          else "нет текстового слоя"),
        "слов слоя не прочитано": v["unread_words"],
        "блоков со словами вне слоя": v["invented"],
        "не найдено обязательных блоков": v["missing"],
    } for v in vision])


def render(cfg: dict, layouts_dir, main_model_prefix: str | None = None) -> None:
    """Весь шаг 4. Базу выбирает dashboard.pick_db: рабочая, если в ней есть
    прогоны, иначе демо-копия (чистый клон ревьюера)."""
    st.subheader("Шаг 4. Мониторинг: как работает система")
    path, is_demo = D.pick_db(cfg)
    con = D.connect(path)
    try:
        data = D.load_all(con, cfg, layouts_dir, main_model_prefix)
    finally:
        con.close()
    s = data["summary"]
    if not data["checks"]:
        st.info("Проверок ещё не было — графики появятся после первого прогона "
                "на шаге 2.")
        return
    if is_demo:
        st.info("Показана **обезличенная демо-копия** журнала (макеты названы "
                "«Макет A, B…», заметки скрыты). Рабочая база появится после "
                "первой проверки.")
    else:
        st.caption(f"Рабочий журнал: {path.name}")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Проверок", s["n_checks"],
              help=f"из них полных прогонов (с вызовами модели): {s['n_full_runs']}")
    k2.metric("Макетов", s["n_layouts"])
    k3.metric("Оценок эксперта", s["n_rated"],
              help="решений и заметок всего: %d" % s["n_feedback"])
    k4.metric("Согласие с системой",
              f"{s['agreement']:.0%}" if s["agreement"] is not None else "—",
              help="доля 👍 среди всех оценок 👍/👎")
    k5.metric("Стоимость, $",
              f"{s['cost_total_usd']:.2f}" if s["cost_total_usd"] is not None else "—",
              help="по прайсу из config.yaml → dashboard.prices_usd_per_1m; "
                   "оценка, не счёт")
    if s["cost_unknown_runs"]:
        st.caption(f"⚠️ У {s['cost_unknown_runs']} прогонов модель без цены в "
                   "конфиге — они в сумму не вошли.")

    st.markdown("#### 1. Статусы вердиктов по прогонам")
    st.caption("Сколько аспектов из 19 регламентных получили каждый статус.")
    st.altair_chart(_status_by_check(data["checks"]), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 2. Проблемные аспекты")
        st.caption("Доля прогонов, где аспект был нарушением или ушёл в ручную "
                   "проверку — по последнему прогону каждого макета. Сверху — "
                   "куда смотреть в первую очередь.")
        if data["aspects"]:
            st.altair_chart(_problem_aspects(data["aspects"]), width="stretch")
        else:
            st.caption("Нет данных по аспектам: для старых прогонов выполните "
                       "`python -m labelcheck.dashboard --backfill`.")
    with c2:
        st.markdown("#### 3. Согласие эксперта с системой")
        st.caption("Оценки 👍/👎 на шаге 2 по аспектам; внизу — где система "
                   "ошибается чаще.")
        if data["ratings"]:
            st.altair_chart(_ratings(data["ratings"]), width="stretch")
        else:
            st.caption("Оценок пока нет.")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### 4. Кому уходят замечания")
        st.caption("Решения по замечаниям на шаге 2 (старые записи журнала "
                   "приведены к этим четырём вариантам).")
        st.altair_chart(_decisions(data["decisions"]), width="stretch")
    with c4:
        st.markdown("#### 5. Стоимость прогонов")
        st.caption("Повтор без правок берёт вердикты из кэша — в журнал "
                   "попадает только план работ, поэтому такие прогоны "
                   "показаны серым, а не «дешёвыми».")
        st.altair_chart(_costs(data["checks"]), width="stretch")

    st.markdown("#### 6. Когда проверяли и сколько это занимало")
    st.altair_chart(_timeline(data["checks"]), width="stretch")

    st.markdown("#### 7. Качество распознавания макетов")
    if data["vision"]:
        st.caption("Блоки макета: сколько прочитано без вмешательства, сколько "
                   "человек подтвердил или поправил на шаге 1, сколько ещё "
                   "помечены сомнительными.")
        st.altair_chart(_vision(data["vision"]), width="stretch")
        st.dataframe(_vision_table(data["vision"]), hide_index=True, width="stretch")
    else:
        st.caption("Нет layout-файлов распознанных макетов — метрики зрения "
                   "появятся после распознавания на шаге 1.")
