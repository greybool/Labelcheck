"""Генерация ground truth для retrieval evaluation (День 7, ТЗ §5.1).

Запуск из корня репозитория:  python evaluation/generate_gt.py [--force]

Схема: стратифицированная выборка чанков (квоты по категориям + пол по
регламентам, seed в конфиге) → CHEAP-модель составляет по каждому чанку
2 вопроса «терминологией фрагмента» (style: verbatim) и 1 «бытовым языком
закупщика» (style: paraphrase) → валидация вопросов КОДОМ (запрет утечки
адресов пунктов и номеров регламентов — иначе BM25 находит пункт по номеру
и метрика завышается) → дедупликация → evaluation/ground_truth.jsonl +
паспорт ground_truth.meta.json с SHA256-пломбой корпуса.

Надёжность: результат каждого чанка сразу пишется в resume-журнал
(ground_truth.partial.jsonl, не в git) — падение API на середине не теряет
оплаченную работу, повторный запуск продолжает с места остановки. Чанк,
из которого модель не смогла составить ни одного валидного вопроса
(«шапка/обрывок»), помечается barren и заменяется резервным чанком той же
страты. Все параметры — labelcheck/config.yaml → eval.
"""

import argparse
import json
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import random

from dotenv import load_dotenv

from evaluation.matcher import category_of, sha256_of, validate_record, load_index
from labelcheck.retrieval import CHUNKS_PATH, load_config

# ── валидация вопросов ───────────────────────────────────────────────────────

# Вопрос бракуется, если содержит адрес нормы или ссылку на «этот фрагмент»:
# такие вопросы либо дают BM25 ответ по номеру (утечка → завышение метрик),
# либо бессмысленны без фрагмента перед глазами.
_BAD_PATTERNS = [re.compile(p, re.IGNORECASE) for p in (
    r"\bтр\s*(тс|еаэс)\b",              # «ТР ТС», «ТР ЕАЭС»
    r"\b0\d{2}\s*/\s*20\d{2}\b",        # «022/2011»
    r"\bпункт\w*\s*№?\s*\d",            # «пункте 14», «пункт № 4»
    r"\bп\.?п?\.\s*\d",                 # «п. 4», «пп. 14»
    r"\bчаст\w+\s*№?\s*\d",             # «части 4.4»
    r"\bстать\w+\s*№?\s*\d",            # «статье 9»
    r"\bприложени\w*\s*№?\s*\d",        # «приложении 2»
    r"\bраздел\w*\s*№?\s*[ivxlc\d]+\b", # «разделе XI» / «разделе 3»
    r"\bподпункт",                      # «в подпунктах „а“–„в“»
    r"\bпункт\w*\s*[«\"'“]",            # «пунктах «а»» — буквенные ссылки
    r"настоящ",                         # «настоящего регламента/документа»
    r"\b(данн|эт)\w+\s+(пункт|фрагмент|документ|регламент|приложени|таблиц)",
)]

# Скрытые ссылки на фрагмент (найдены разбором промахов v1): вопрос со
# словами «этот/такой/здесь» без названного объекта, вопросы про оформление
# таблицы («пометка н/д», «графа») и про внутренние каталожные номера —
# неотвечаемы без фрагмента перед глазами, поиску их предъявлять нечестно.
# Идиома «при этом» вырезается до проверки — она ссылкой не является.
_IDIOM_RE = re.compile(r"\bпри этом\b", re.IGNORECASE)
_DEICTIC_PATTERNS = [re.compile(p, re.IGNORECASE) for p in (
    r"\bэт(?:о|от|а|и|у|ой|ом|им|ими|их)\b",   # «этот ингредиент», «из этой группы»
    r"\bздесь\b",
    r"\bтак(?:ой|ая|ое|ие|ую)\s+продукци",     # «в такой продукции»
    r"\b(?:в|из)\s+(?:данн\w+\s+)?(?:таблиц|списк|перечн)",  # «в таблице», «из списка»
    r"\bпометк", r"\bн/д\b", r"\bграф[аеы]\b",  # вопросы про оформление таблицы
    r"\bкаталог", r"\bхимическ\w+\s+баз",       # «номер в химической базе»
    r"\bс\s+номером\s+\d",                      # внутренние номера перечней
)]


def question_problems(q: str) -> list[str]:
    """Проверка одного вопроса. Пустой список = вопрос годен."""
    problems = []
    if not isinstance(q, str) or len(q.strip()) < 15:
        return ["слишком короткий или не строка"]
    if len(q) > 300:
        problems.append("длиннее 300 символов")
    if "?" not in q:
        problems.append("нет вопросительного знака")
    for pat in _BAD_PATTERNS:
        if pat.search(q):
            problems.append(f"утечка адреса/ссылка на фрагмент: {pat.pattern!r}")
    cleaned = _IDIOM_RE.sub(" ", q)
    for pat in _DEICTIC_PATTERNS:
        if pat.search(cleaned):
            problems.append(f"вопрос не самодостаточен: {pat.pattern!r}")
    return problems


def normalize_question(q: str) -> str:
    """Ключ дедупликации: регистр, ё→е, только буквы/цифры, одиночные пробелы."""
    q = q.lower().replace("ё", "е")
    q = re.sub(r"[^а-яa-z0-9 ]", " ", q)
    return re.sub(r"\s+", " ", q).strip()


# ── явное поле accepted_chunk_ids ────────────────────────────────────────────

def part_run(pos: int, chunks: list[dict]) -> list[str]:
    """Все part-части того же пункта, что и чанк chunks[pos] (включая его).

    Части одного длинного пункта лежат в chunks.jsonl подряд, с одинаковыми
    метаданными адреса и part = 1, 2, … Разные пункты с одинаковым номером
    внутри приложения (нумерация там перезапускается) в один run не попадут:
    их part-цепочки не смежны. Для чанка без part — только он сам.
    """
    me = chunks[pos]
    if not me.get("part"):
        return [me["chunk_id"]]

    def same_clause(c: dict) -> bool:
        return all(c.get(f) == me.get(f) for f in
                   ("regulation_id", "appendix", "section", "subsection", "clause"))

    start = pos
    while (start > 0 and same_clause(chunks[start - 1])
           and chunks[start - 1].get("part") == chunks[start]["part"] - 1):
        start -= 1
    end = pos
    while (end + 1 < len(chunks) and same_clause(chunks[end + 1])
           and chunks[end + 1].get("part") == chunks[end]["part"] + 1):
        end += 1
    return [c["chunk_id"] for c in chunks[start:end + 1]]


# ── стратифицированная выборка ───────────────────────────────────────────────

def allocate(quota: int, avail: dict[str, int], floor: int) -> dict[str, int]:
    """Распределение квоты категории по регламентам: пол + пропорция.

    Каждый регламент получает минимум floor чанков (или все, если их меньше);
    остаток квоты делится пропорционально оставшейся доступности (метод
    наибольших остатков, при равенстве — по алфавиту regulation_id, чтобы
    результат был детерминирован). Если полов больше квоты — полы урезаются
    по одному с конца алфавита, тоже детерминированно.
    """
    regs = sorted(avail)
    alloc = {r: min(floor, avail[r]) for r in regs}
    while sum(alloc.values()) > quota:
        for r in reversed(regs):
            if alloc[r] > 0:
                alloc[r] -= 1
                break
    rest = quota - sum(alloc.values())
    room = {r: avail[r] - alloc[r] for r in regs}
    total_room = sum(room.values())
    if rest > 0 and total_room > 0:
        shares = {r: rest * room[r] / total_room for r in regs}
        for r in regs:
            take = min(int(shares[r]), room[r])
            alloc[r] += take
            room[r] -= take
        rest = quota - sum(alloc.values())
        by_frac = sorted(regs, key=lambda r: (-(shares[r] - int(shares[r])), r))
        for r in by_frac:
            if rest == 0:
                break
            if room[r] > 0:
                alloc[r] += 1
                room[r] -= 1
                rest -= 1
    return alloc


def build_sample(chunks: list[dict], ecfg: dict):
    """Выборка чанков и резерв для замен. Детерминирована seed'ом.

    Возвращает (selected, reserves): selected — список позиций чанков;
    reserves — {(category, regulation): [позиции…]} из невыбранных кандидатов,
    откуда берутся замены barren-чанков (сначала тот же регламент,
    потом любой в категории).
    """
    rng = random.Random(ecfg["seed"])
    min_chars = ecfg["min_chunk_chars"]

    pool = defaultdict(list)  # (category, regulation) -> [позиции]
    for pos, c in enumerate(chunks):
        if len(c["text"]) >= min_chars:
            pool[(category_of(c), c["regulation_id"])].append(pos)

    selected, reserves = [], {}
    for cat, quota in ecfg["quotas"].items():
        avail = {reg: len(pool[(c, reg)]) for (c, reg) in pool if c == cat}
        if not avail:
            continue
        alloc = allocate(quota, avail, ecfg["min_per_regulation"])
        for reg in sorted(avail):
            candidates = sorted(pool[(cat, reg)])
            take = rng.sample(candidates, alloc[reg])
            selected.extend(take)
            rest = [p for p in candidates if p not in set(take)]
            rng.shuffle(rest)
            reserves[(cat, reg)] = rest
    return sorted(selected), reserves


# ── генерация вопросов по чанку ──────────────────────────────────────────────

def parse_questions(raw: str, n_verbatim: int, n_paraphrase: int) -> dict:
    """Ответ модели → {"verbatim": [...], "paraphrase": [...]} (с обрезкой)."""
    data = json.loads(raw)
    out = {}
    for style, n in (("verbatim", n_verbatim), ("paraphrase", n_paraphrase)):
        items = data.get(style, [])
        if not isinstance(items, list):
            raise ValueError(f"{style}: не список")
        out[style] = [q.strip() for q in items if isinstance(q, str) and q.strip()][:n]
    return out


def generate_for_chunk(client, model: str, prompt: str, pos: int,
                       chunks: list[dict], ecfg: dict) -> dict:
    """Вопросы по одному чанку. Возвращает строку resume-журнала.

    До 3 попыток вызова (сетевые сбои и битый JSON — с паузой);
    негодные вопросы отбраковываются валидатором, их причины — в rejected.
    """
    chunk = chunks[pos]
    accepted = part_run(pos, chunks)
    last_err = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": prompt},
                          {"role": "user", "content": chunk["text"]}])
            parsed = parse_questions(resp.choices[0].message.content,
                                     ecfg["questions_verbatim"],
                                     ecfg["questions_paraphrase"])
            break
        except Exception as e:  # noqa: BLE001 — ретраи с паузой, потом наверх
            last_err = e
            time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"{chunk['chunk_id']}: 3 неудачные попытки: {last_err}")

    records, rejected = [], []
    for style, questions in parsed.items():
        for q in questions:
            problems = question_problems(q)
            if problems:
                rejected.append({"question": q, "problems": problems})
                continue
            records.append({
                "question": q,
                "chunk_id": chunk["chunk_id"],
                "regulation_id": chunk["regulation_id"],
                "category": category_of(chunk),
                "style": style,
                "accepted_chunk_ids": accepted,
            })
    usage = resp.usage
    return {"chunk_id": chunk["chunk_id"], "pos": pos,
            "barren": not records, "records": records, "rejected": rejected,
            "tokens": {"prompt": usage.prompt_tokens,
                       "completion": usage.completion_tokens}}


# ── оркестрация ──────────────────────────────────────────────────────────────

def load_partial(path: Path) -> dict[str, dict]:
    """Resume-журнал → {chunk_id: строка журнала}."""
    done = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                done[row["chunk_id"]] = row
    return done


def run_generation(client, chunks, ecfg, model, prompt, partial_path):
    """Параллельная генерация с resume и заменой barren-чанков."""
    selected, reserves = build_sample(chunks, ecfg)
    done = load_partial(partial_path)
    lock = threading.Lock()

    def next_replacement(cat: str, reg: str):
        with lock:
            queue = reserves.get((cat, reg), [])
            if queue:
                return queue.pop(0)
            for (c, r), q in sorted(reserves.items()):
                if c == cat and q:
                    return q.pop(0)
        return None

    pending = [p for p in selected if chunks[p]["chunk_id"] not in done]
    print(f"Выборка: {len(selected)} чанков, уже готово: "
          f"{len(selected) - len(pending)}, к генерации: {len(pending)}")

    replacements = 0
    with open(partial_path, "a", encoding="utf-8") as journal, \
            ThreadPoolExecutor(max_workers=ecfg["concurrency"]) as ex:
        futures = {ex.submit(generate_for_chunk, client, model, prompt,
                             p, chunks, ecfg): p for p in pending}
        while futures:
            for fut in as_completed(list(futures)):
                pos = futures.pop(fut)
                row = fut.result()  # исключение = стоп всего прогона (громко)
                with lock:
                    journal.write(json.dumps(row, ensure_ascii=False) + "\n")
                    journal.flush()
                    done[row["chunk_id"]] = row
                if row["barren"]:
                    chunk = chunks[pos]
                    sub = next_replacement(category_of(chunk),
                                           chunk["regulation_id"])
                    if sub is not None and chunks[sub]["chunk_id"] not in done:
                        replacements += 1
                        futures[ex.submit(generate_for_chunk, client, model,
                                          prompt, sub, chunks, ecfg)] = sub
                n_q = sum(len(r["records"]) for r in done.values())
                print(f"\r  чанков: {len(done)}, вопросов: {n_q}, "
                      f"замен barren: {replacements}", end="", flush=True)
    print()
    return done


def finalize(done: dict, chunks, ecfg, model, prompt, corpus_sha,
             gt_path: Path, meta_path: Path):
    """Дедуп, сквозная валидация, детерминированная запись эталона и паспорта."""
    order = {c["chunk_id"]: i for i, c in enumerate(chunks)}
    records = [r for row in done.values() for r in row["records"]]
    # Страховка: валидатор применяется и здесь — правило, добавленное после
    # генерации, вычищает старые записи из журнала при повторной финализации.
    finalize_rejected = [r for r in records if question_problems(r["question"])]
    records = [r for r in records if not question_problems(r["question"])]
    records.sort(key=lambda r: (order[r["chunk_id"]], r["style"], r["question"]))

    seen, unique, duplicates = set(), [], 0
    for r in records:
        key = normalize_question(r["question"])
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append(r)

    index = load_index(CHUNKS_PATH)
    for r in unique:
        problems = validate_record(r, index)
        if problems:
            raise ValueError(f"битая запись {r['chunk_id']}: {problems}")

    with open(gt_path, "w", encoding="utf-8") as f:
        for r in unique:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    import hashlib
    from datetime import date
    tokens = Counter()
    for row in done.values():
        tokens.update(row["tokens"])
    meta = {
        "corpus_sha256": corpus_sha,
        "model": model,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16],
        "seed": ecfg["seed"],
        "quotas": ecfg["quotas"],
        "min_per_regulation": ecfg["min_per_regulation"],
        "generated_at": date.today().isoformat(),
        "n_questions": len(unique),
        "by_category": dict(Counter(r["category"] for r in unique)),
        "by_style": dict(Counter(r["style"] for r in unique)),
        "by_regulation": dict(Counter(r["regulation_id"] for r in unique)),
        "chunks_used": sum(1 for row in done.values() if not row["barren"]),
        "chunks_barren": sum(1 for row in done.values() if row["barren"]),
        "dropped_duplicates": duplicates,
        "rejected_questions": sum(len(row["rejected"]) for row in done.values()),
        "rejected_at_finalize": len(finalize_rejected),
        "tokens": dict(tokens),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="перезаписать существующий эталон")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    cfg = load_config()
    ecfg = cfg["eval"]
    gt_path = ROOT / ecfg["gt_path"]
    meta_path = ROOT / ecfg["gt_meta_path"]
    partial_path = ROOT / ecfg["gt_partial_path"]

    if gt_path.exists() and not args.force:
        print(f"{gt_path} уже существует — прогоны метрик могли на него "
              "опираться. Перегенерация: --force")
        return 1

    import os

    from openai import OpenAI
    model = os.environ[ecfg["gt_model_env"]]
    prompt = ecfg["gt_prompt"].format(n_verbatim=ecfg["questions_verbatim"],
                                      n_paraphrase=ecfg["questions_paraphrase"])
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    corpus_sha = sha256_of(CHUNKS_PATH)

    client = OpenAI()
    done = run_generation(client, chunks, ecfg, model, prompt, partial_path)
    meta = finalize(done, chunks, ecfg, model, prompt, corpus_sha,
                    gt_path, meta_path)

    print(f"Эталон: {meta['n_questions']} вопросов → {gt_path}")
    print(f"  по категориям: {meta['by_category']}")
    print(f"  по стилям:     {meta['by_style']}")
    print(f"  по регламентам:{meta['by_regulation']}")
    print(f"  barren-чанков: {meta['chunks_barren']}, дублей снято: "
          f"{meta['dropped_duplicates']}, забраковано валидатором: "
          f"{meta['rejected_questions']}")
    print(f"  токены CHEAP:  {meta['tokens']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
