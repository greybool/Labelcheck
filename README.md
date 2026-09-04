# LabelCheck — pre-flight check of food packaging layouts against EAEU technical regulations

*Final project for LLM Zoomcamp 2026 (DataTalksClub). Русская версия — [ниже](#labelcheck-по-русски).*

> LabelCheck is a **pre-check tool**. The final decision on any label is made by a
> qualified specialist and a lawyer — the app says so on every screen.

## 1. The problem

A procurement or quality specialist at a food importer receives a packaging layout
(a designer's PDF) from a foreign supplier before a production run. The label has
to comply with the EAEU technical regulations — the horizontal ones
(TR CU 022/2011 on labelling, 021/2011 on food safety, 029/2012 on additives,
005/2011 on packaging) plus a category-specific one (meat, poultry, fish). The
requirements are spread over seven documents and their appendix tables; checking
one layout by hand takes hours, and a miss is expensive: a reprinted print run or a
shipment held at customs.

LabelCheck reads the layout with a vision model, splits it into text blocks, lets
the human correct the recognised text, and then checks **21 aspects** (product
name, composition, allergens, additives, net weight, nutrition, dates, storage,
manufacturer, importer, GMO, EAC mark, packaging marks, language, font size,
warning labels, claims, barcode, spelling…) against the regulations. Every verdict
is *compliant / possible violation / needs manual review* and must quote the exact
clause it relies on — a verdict without a valid quotation is downgraded to
"needs manual review" by code, never by the model.

The knowledge base is the **text of the regulations themselves** — seven official
consolidated editions (PDF, in `data/raw/`, with a SHA-256 manifest in
`data/sources.yaml`). No course FAQ documents are used.

## 2. How it works

```
7 regulation PDFs ─► pdfplumber ─► cleanup rules (YAML) ─► 2 417 chunks (1 chunk = 1 clause)
                                                                │
                              ┌─────────────────────────────────┴───────────────┐
                              ▼                                                 ▼
                        BM25 (in memory)                          OpenAI text-embedding-3-small
                                                                  → Qdrant (memory or server)
                              └──────────────── hybrid RRF ─────────────────────┘
                                                      ▲ query rewriting (cheap model)
Packaging PDF ─► render ─► vision pass 1: region map ─► pass 2: verbatim read per crop
             ─► text-layer guards (typos, substitutions, hallucinations) ─► human edits
             ─► 21 aspects: basis clauses + retrieval + E-code lookups ─► verdicts with quotes
             ─► code validation of every quote ─► report ─► work plan ─► SQLite journal ─► dashboard
```

Key design decisions (details and measurements in [`docs/PIPELINE.md`](docs/PIPELINE.md)):

* **Chunk = clause.** A verdict must cite a clause, so the retrieval unit equals the
  citation unit. Appendix tables are cut into overlapping windows with a repeated
  header. Chunk metadata: regulation, section, subsection, clause, appendix.
* **Hybrid search + query rewriting.** BM25 (Snowball stemming, E-code
  normalisation) and vectors are fused by Reciprocal Rank Fusion; a cheap model
  rewrites the query "in the language of the regulation". Both are configurable
  and both are measured separately (see §3).
* **Basis-first context.** Each aspect in [`labelcheck/aspects.yaml`](labelcheck/aspects.yaml)
  names the clauses it is grounded in; those clauses always go into the prompt,
  retrieval adds the rest. A test fails if a configured clause address does not
  resolve in the corpus.
* **Two-pass vision.** One call over the whole layout hallucinated ingredients
  (allergens "spinach", "peanuts" that were not on the label) because the API
  downscales large images. The overview pass maps regions on a downscaled image;
  the reading pass transcribes full-resolution crops. Where the PDF has a text
  layer it is used as a **guard, not as input**: unread words, invented words,
  silent word substitutions ("молодой" → "молотый"), invented barcodes.
* **Anti-hallucination by code.** Quotations must be verbatim substrings of the
  passed clauses; clause numbers and °C/% figures in the explanation must appear in
  the citations or the label facts; unstable aspects are voted 3×.
* **Human in the loop.** Step 1 shows every block with its crop and lets the user
  fix or confirm the text; step 2 records 👍/👎 and a decision (designer /
  supplier / check myself) per verdict; step 3 turns it into a work plan.

## 3. Evaluation

### Retrieval (618 questions, all five metrics)

Ground truth: 618 questions generated for 240 stratified chunks (2 verbatim + 1
paraphrase per chunk, address leaks rejected by code), sealed to the corpus hash.
Scoring rules and the manual analysis of all 44 hybrid misses are in
[`evaluation/EVALUATION.md`](evaluation/EVALUATION.md); raw runs in
`evaluation/runs/`, metrics in `evaluation/metrics/`.

| method | rewriting | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|---|---|
| BM25 | off | 0.626 | 0.796 | 0.850 | 0.888 | 0.720 |
| BM25 | on | 0.647 | 0.822 | 0.869 | 0.909 | 0.743 |
| vector | off | 0.579 | 0.749 | 0.811 | 0.854 | 0.672 |
| vector | on | 0.597 | 0.754 | 0.817 | 0.861 | 0.686 |
| hybrid RRF | off | 0.663 | 0.827 | 0.885 | 0.953 | 0.762 |
| **hybrid RRF** | **on** | **0.686** | **0.845** | **0.904** | 0.952 | **0.777** |

Rewriting helps exactly where it should: on paraphrased (buyer's-language)
questions hybrid Hit@5 goes 0.793 → 0.869; on verbatim questions it is neutral.
Breakdowns by question style, chunk category and regulation are in
`evaluation/metrics/summary.md`.

### LLM evaluation

* **LLM-as-judge** (cheap model, different from the verdict model) in two modes —
  *strict* (sees only what the report reader sees) and *with basis* (sees the
  full context). The judge's own noise was measured (mode agreement 67%,
  self-consistency 6/12) and is documented as a limit of the method. After the
  Day-9 fixes: reasoning supported 14/18, invented clauses 1/18, citations
  relevant 18/18 on the control layout.
* **Stability** (3 full runs of one layout, cache off): 68% → 89% (voting) →
  **95%** of aspects keep their status.
* **Cosine similarity ($0)** between the target clause and the top-1 hit: 0.915
  on average; 0.926 on hits vs 0.700 on misses — misses are "neighbours", not
  random clauses.
* **Vision ground truth** (30 fields on 3 real layouts, confirmed by the domain
  expert): 67% exact in the first version → **90.3%** after the vision package.

### Tests

12 suites, **268 checks**, no API calls: `for t in tests/test_*.py; do python $t; done`.
They cover the parser (domain regressions), the matcher, retrieval, vision
geometry and guards, aspects (every configured clause resolves), verdict
validation, the store, the dashboard normalisation, the demo-DB anonymiser and
repository hygiene.

## 4. Interface and monitoring

`streamlit run labelcheck/app.py` — four steps:

1. **Layout** — upload a PDF (vision, ~$0.10) or open a recognised one; every block
   with its crop, editable text, a full diff against the PDF text layer, confirm /
   fix buttons.
2. **Check** — base regulations always; meat / poultry / fish switched on by the
   human; 21 verdicts with quotations, votes and downgrade reasons; per verdict a
   decision and a 👍/👎 rating, autosaved to SQLite.
3. **Work plan** — short "what to do" items for the designer, the supplier and
   yourself; Markdown and Word export.
4. **Monitoring** — seven charts over the journal: verdict statuses per run,
   problem aspects, expert agreement (👍/👎) per aspect, decision targets, cost per
   run, timeline, recognition quality. On a fresh clone the dashboard shows an
   anonymised demo journal (`data/labelcheck.demo.db`, built by
   `evaluation/make_demo_db.py`).

## 5. Run it

### Prerequisites

* An OpenAI API key. Costs: vector cache once ≈ $0.03; reading one layout ≈
  $0.10; a full check of one layout ≈ $1 (a repeat without edits is cached and free).
* Docker (recommended) **or** Python 3.12.

```bash
git clone https://github.com/greybool/Labelcheck.git && cd Labelcheck
cp .env.example .env        # put your OPENAI_API_KEY here; model names are pre-filled
```

### With Docker (Qdrant server + app)

```bash
docker compose up --build -d                     # http://localhost:8501
docker compose run --rm app python ingestion/index.py   # once: vector cache + Qdrant collection
```

`data/` is bind-mounted, so the vector cache, the journal, uploaded PDFs and
recognised layouts survive rebuilds. The app is switched to the Qdrant server by
`QDRANT_MODE`/`QDRANT_URL` in `docker-compose.yml`; the config file keeps the
in-memory mode for running without Docker.

### Without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python ingestion/index.py            # once: 2 417 embeddings → data/embeddings.npz (~$0.03)
streamlit run labelcheck/app.py      # http://localhost:8501
```

### Try it

Upload `data/samples/demo_label.pdf` on step 1 — a synthetic frozen-berry label
(generated by `evaluation/make_demo_label.py`, no real product) with a few planted
defects: a Latin "E" in an E-code, a misspelt word, an unsupported "GMO-free"
claim. Real layouts used during development are private and are not in the
repository.

CLI without the UI: `python -m labelcheck.check data/samples/demo_label.pdf`
(reports go to `data/reports/`).

### Reproduce the evaluation

```bash
python evaluation/run_retrieval_eval.py --metrics-only   # metrics from the committed runs, no API
python evaluation/run_retrieval_eval.py                  # re-run all 6 configurations (API)
python -m labelcheck.check <layout.json> --no-cache -o data/reports/runN   # ×3, then:
python evaluation/stability.py data/reports/run1/*.json data/reports/run2/*.json data/reports/run3/*.json
python evaluation/judge.py <report.json> --layout <layout.json> [--with-basis]
python ingestion/parse.py                                # rebuild chunks from the PDFs (~5 min)
python data/verify_corpus.py                             # SHA-256 check of the corpus
```

## 6. Repository map

```
labelcheck/      app.py (UI) · check.py (CLI) · vision.py · verdict.py · actions.py (work plan)
                 retrieval.py · rewrite.py · aspects.py + aspects.yaml (21 aspects) · config.yaml
                 store.py (SQLite) · dashboard.py + dashboard_ui.py (monitoring)
ingestion/       parse.py (PDF → chunks) · cleanup.yaml (rules) · index.py (embeddings → Qdrant)
evaluation/      ground truth, retrieval metrics, judge, stability, cosine, vision GT,
                 make_demo_db.py, make_demo_label.py, EVALUATION.md
data/            raw/ (7 regulation PDFs + manifest) · chunks.jsonl · query_rewrites.json
                 labelcheck.demo.db · samples/demo_label.pdf
tests/           12 suites, 268 checks
docs/            PIPELINE.md (what is built, with numbers) · REVIEW-LOG.md (acceptance
                 findings R-01…R-40 with root causes) · TZ-LABELCHECK.md (spec) · WORKING-PRACTICES.md
Dockerfile · docker-compose.yml · requirements.txt · .env.example
```

All tunables — thresholds, model tiers, prices, voting, guards — live in
`labelcheck/config.yaml` and `.env`, not in code.

## 7. Honest limitations

* Vision reading is non-deterministic between runs of the same PDF (coverage of
  the text layer varied 98% → 82% on one layout); the app shows unread words and
  asks the human to re-read or fix — automatic re-reading is on the backlog.
* Pictograms without text (the Mobius loop) are often missed by the overview pass.
* The cheap LLM judge is noisy; its numbers are reported with their measured
  disagreement, not as absolute truth.
* Claims arithmetic (appendix 5 of TR CU 022 — "source of protein" needs two
  conditions) is not computed yet; the aspect reports what it can quote.
* Regulation editions are consolidated texts from legal databases, pinned by
  hash; official EEC PDFs for two regulations are scans and were not usable.

Everything above is tracked in [`docs/REVIEW-LOG.md`](docs/REVIEW-LOG.md) and
[`BACKLOG.md`](BACKLOG.md).

---

## LabelCheck по-русски

**Что это.** Инструмент предварительной проверки макета упаковки пищевой
продукции на соответствие техническим регламентам ЕАЭС (ТР ТС 022/2011,
021/2011, 029/2012, 005/2011 + профильные 034/2013 мясо, 051/2021 птица,
040/2016 рыба). Пользователь — специалист по закупкам или качеству, который
получает PDF-макет от иностранного поставщика. Финальное решение — за
специалистом и юристом; это написано в интерфейсе.

**Как устроено.** Корпус — тексты семи регламентов, нарезанные по принципу
«один чанк = один пункт» (2 417 чанков). Поиск — гибрид BM25 + векторы
(text-embedding-3-small, Qdrant) со слиянием RRF и переформулировкой запроса
дешёвой моделью. Макет читается vision-моделью в два прохода (карта блоков →
дословное чтение кропов полного разрешения), текстовый слой PDF используется
как сторож: непрочитанные слова, слова вне слоя, подмены («молодой» →
«молотый»), выдуманные штрихкоды. 21 аспект проверки описан в
`labelcheck/aspects.yaml` с адресами пунктов-оснований; вердикт обязан
процитировать пункт дословно — цитаты проверяет код, без валидной цитаты статус
понижается до «требует ручной проверки». Человек правит распознанный текст
до проверки, ставит 👍/👎 и решение по каждому замечанию; система собирает план
работ для дизайнера, поставщика и самого проверяющего.

**Метрики.** Retrieval на 618 вопросах: гибрид + rewriting — Hit@1 0.686,
Hit@3 0.845, Hit@5 0.904, Hit@10 0.952, MRR 0.777 (таблица всех шести
конфигураций выше). LLM-оценка: судья в двух режимах (с измеренным шумом
самого судьи), стабильность вердиктов 95%, cosine-метрика 0.915, vision-эталон
90,3% точных полей. Тесты: 12 наборов, 268 проверок, без API.

**Запуск.** `cp .env.example .env` (ключ OpenAI) → `docker compose up --build -d`
→ один раз `docker compose run --rm app python ingestion/index.py` (кэш векторов,
≈ $0,03) → http://localhost:8501. Без докера: `pip install -r requirements.txt`,
`python ingestion/index.py`, `streamlit run labelcheck/app.py`. Попробовать —
на синтетическом макете `data/samples/demo_label.pdf` (вымышленный продукт с
подложенными дефектами). Проверка одного макета ≈ $1.

**Что где.** `docs/PIPELINE.md` — карта всего построенного с цифрами;
`docs/REVIEW-LOG.md` — журнал приёмки владельцем: 40 замечаний с разбором по
сырым данным и статусами; `BACKLOG.md` — отложенные идеи;
`evaluation/EVALUATION.md` — правила зачёта и разбор промахов.
