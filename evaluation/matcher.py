"""Матчер эталона: решает, засчитывать ли найденный чанк как попадание.

Используется в retrieval evaluation (День 7): ground truth задаёт для каждого
вопроса правильный чанк-источник, поиск возвращает список chunk_id — матчер
определяет ранг правильного ответа в этом списке. Из рангов дальше считаются
Hit@1/3/5/10 и MRR.

Правила попадания (обоснование: evaluation/EVALUATION.md):
- пункт из ТЕЛА документа: попадание = любой чанк с тем же адресом
  (регламент + раздел + подраздел + пункт). Длинный пункт разрезан на части
  с одинаковыми метаданными — все части дают одну и ту же цитату,
  поэтому равнозначны. Уникальность адреса в теле закреплена тестом
  test_body_clause_address_unique.
- пункт из ПРИЛОЖЕНИЯ, окно таблицы, текст без номера: попадание = точно
  тот же chunk_id. В приложениях нумерация пунктов перезапускается
  в каждой внутренней таблице, адрес неуникален; у окон номеров нет.

Защита от рассинхронизации: chunk_id позиционные, любой перепарсинг корпуса
их сдвигает. Загрузка ground truth сверяет SHA256 текущего chunks.jsonl
с хэшем, записанным при генерации вопросов, и падает при расхождении.
"""

import hashlib
import json
from pathlib import Path

# Категории вопросов. У каждой своё правило зачёта и своя строка
# в таблице метрик (разбивка по категориям, не одно число).
CATEGORIES = ("body_clause", "appendix_clause", "table_window", "no_clause")

# Стили вопросов эталона (День 7): verbatim — терминологией фрагмента,
# paraphrase — бытовым языком закупщика. Разбивка метрик по стилю показывает
# эффект query rewriting: на verbatim он ≈ нулевой (вопрос уже «языком
# регламента»), на paraphrase — виден.
STYLES = ("verbatim", "paraphrase")


def sha256_of(path: Path) -> str:
    """SHA256-хэш файла, читая кусками по 1 МБ (как в data/verify_corpus.py)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def category_of(chunk: dict) -> str:
    """Категория чанка по его явным полям метаданных (без эвристик)."""
    if chunk.get("is_table"):
        return "table_window"
    if chunk.get("clause"):
        return "appendix_clause" if chunk.get("appendix") else "body_clause"
    return "no_clause"


def match_key(chunk: dict):
    """Ключ, по которому считается попадание.

    Для пункта из тела — адрес пункта (все part-части одного пункта получают
    одинаковый ключ). Для всех остальных — сам chunk_id: точнее сравнивать
    не с чем. Первый элемент кортежа различает типы ключей, чтобы пункт
    тела случайно не совпал с чанком приложения.
    """
    if category_of(chunk) == "body_clause":
        return ("clause", chunk["regulation_id"], chunk.get("section"),
                chunk.get("subsection"), chunk["clause"])
    return ("chunk", chunk["chunk_id"])


def load_index(chunks_path: Path) -> dict[str, dict]:
    """chunks.jsonl → словарь chunk_id → {category, key, regulation_id}.

    Текст чанков матчеру не нужен, поэтому в память берутся только метаданные.
    """
    index = {}
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            index[c["chunk_id"]] = {"category": category_of(c),
                                    "key": match_key(c),
                                    "regulation_id": c["regulation_id"]}
    return index


def is_hit(gold_id: str, retrieved_id: str, index: dict) -> bool:
    """Считается ли найденный чанк попаданием в правильный ответ.

    Неизвестный chunk_id с любой стороны — ошибка, а не молчаливый промах:
    это всегда признак рассинхронизации корпуса, эталона или индекса.
    """
    for name, cid in (("эталона", gold_id), ("поиска", retrieved_id)):
        if cid not in index:
            raise ValueError(f"неизвестный chunk_id из {name}: {cid!r} — "
                             "корпус и данные оценки рассинхронизированы")
    return index[gold_id]["key"] == index[retrieved_id]["key"]


def rank_of(gold_id: str, retrieved_ids: list[str], index: dict):
    """Позиция первого попадания в выдаче (1 = первое место) или None.

    Из ранга считаются метрики: Hit@K — попал ли ранг в первые K,
    MRR — среднее значение 1/ранг (0, если попадания нет вовсе).
    """
    for pos, rid in enumerate(retrieved_ids, start=1):
        if is_hit(gold_id, rid, index):
            return pos
    return None


def rank_of_record(rec: dict, retrieved_ids: list[str], index: dict):
    """Ранг попадания для записи эталона (расширение rank_of, День 7).

    Попадание = совпадение match_key (для пунктов тела это адрес — любая
    part-часть равнозначна) ИЛИ chunk_id из явного списка
    rec["accepted_chunk_ids"]. Список заполняется при генерации эталона
    детерминированно (part-части того же пункта приложения) — правило зачёта
    остаётся явным полем схемы, без эвристик в eval-коде (ТЗ §5).
    """
    accepted = set(rec.get("accepted_chunk_ids") or [rec["chunk_id"]])
    for pos, rid in enumerate(retrieved_ids, start=1):
        if is_hit(rec["chunk_id"], rid, index) or rid in accepted:
            return pos
    return None


def validate_record(rec: dict, index: dict) -> list[str]:
    """Проверка одной строки ground_truth.jsonl. Возвращает список проблем.

    Пустой список = запись корректна. Используется при генерации эталона
    (День 7) и в тестах — битые записи не должны попадать в метрики молча.
    """
    problems = []
    for field in ("question", "chunk_id", "regulation_id", "category"):
        if not rec.get(field):
            problems.append(f"пустое поле {field}")
    if problems:
        return problems
    chunk = index.get(rec["chunk_id"])
    if chunk is None:
        return [f"chunk_id {rec['chunk_id']!r} нет в корпусе"]
    if rec["category"] != chunk["category"]:
        problems.append(f"категория {rec['category']!r} не совпадает "
                        f"с категорией чанка {chunk['category']!r}")
    if rec["regulation_id"] != chunk["regulation_id"]:
        problems.append(f"regulation_id {rec['regulation_id']!r} не совпадает "
                        f"с регламентом чанка {chunk['regulation_id']!r}")
    # Поля Дня 7 — необязательные для старых записей, но если есть, то валидные.
    if "style" in rec and rec["style"] not in STYLES:
        problems.append(f"неизвестный style {rec['style']!r}")
    if "accepted_chunk_ids" in rec:
        acc = rec["accepted_chunk_ids"]
        if not isinstance(acc, list) or rec["chunk_id"] not in acc:
            problems.append("accepted_chunk_ids обязан быть списком "
                            "и содержать сам chunk_id")
        else:
            unknown = [cid for cid in acc if cid not in index]
            if unknown:
                problems.append(f"accepted_chunk_ids вне корпуса: {unknown!r}")
    return problems


def load_ground_truth(gt_path: Path, meta_path: Path,
                      chunks_path: Path) -> tuple[list[dict], dict]:
    """Загружает эталон, сверив его с текущим корпусом. Возвращает (записи, индекс).

    Порядок: 1) SHA256 текущего chunks.jsonl обязан совпадать с хэшем из
    meta-файла эталона (chunk_id позиционные — после перепарсинга старый
    эталон недействителен и должен падать громко, а не искажать метрики);
    2) каждая запись проходит validate_record.
    """
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    actual = sha256_of(Path(chunks_path))
    if actual != meta["corpus_sha256"]:
        raise ValueError(
            "ground truth сгенерирован для другой версии корпуса:\n"
            f"  в meta:  {meta['corpus_sha256']}\n"
            f"  сейчас:  {actual}\n"
            "Перегенерируй эталон (evaluation/ground_truth.py) после "
            "изменения chunks.jsonl.")

    index = load_index(chunks_path)
    records = []
    with open(gt_path, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            rec = json.loads(line)
            problems = validate_record(rec, index)
            if problems:
                raise ValueError(f"ground_truth.jsonl, строка {n}: "
                                 + "; ".join(problems))
            records.append(rec)
    return records, index
