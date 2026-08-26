"""Тесты матчера эталона (evaluation/matcher.py).

Негативные кейсы обязательны: матчер, который завышает (ставит зачёт там,
где его нет), опаснее заниженного — красивые метрики скрыли бы плохой поиск.

Запуск из корня репозитория:  python tests/test_matcher.py
(совместим и с pytest: pytest tests/)
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.matcher import (CATEGORIES, category_of, is_hit, load_index,
                                load_ground_truth, match_key, rank_of,
                                sha256_of, validate_record)

CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"

with open(CHUNKS_PATH, encoding="utf-8") as f:
    CHUNKS = [json.loads(line) for line in f]
INDEX = load_index(CHUNKS_PATH)


def find_id(**conditions) -> str:
    """chunk_id первого чанка, у которого все поля равны указанным значениям."""
    for c in CHUNKS:
        if all(c.get(k) == v for k, v in conditions.items()):
            return c["chunk_id"]
    raise AssertionError(f"в корпусе нет чанка с полями {conditions}")


# --- Позитивные кейсы: где зачёт ОБЯЗАН ставиться ---

def test_part_split_is_hit():
    """Части одного разрезанного пункта тела равнозначны: вопрос из части 1,
    найдена часть 2 → попадание (цитата у частей одна и та же)."""
    groups = {}
    for c in CHUNKS:
        if category_of(c) == "body_clause" and c.get("part"):
            groups.setdefault(match_key(c), []).append(c["chunk_id"])
    split = [ids for ids in groups.values() if len(ids) > 1]
    assert split, "в корпусе не осталось part-разрезанных пунктов тела"
    parts = split[0]
    assert is_hit(parts[0], parts[1], INDEX)
    assert is_hit(parts[1], parts[0], INDEX)


def test_exact_chunk_is_hit():
    """Любой чанк — попадание сам для себя (все категории)."""
    for cat in CATEGORIES:
        ids = [c["chunk_id"] for c in CHUNKS if category_of(c) == cat]
        assert ids, f"в корпусе нет чанков категории {cat}"
        assert is_hit(ids[0], ids[0], INDEX), f"категория {cat}"


def test_rank_positions():
    """rank_of: позиция первого попадания, 1-базовая; None — если попадания нет."""
    gold = find_id(regulation_id="ТР ТС 022/2011", clause="1")
    other = find_id(regulation_id="ТР ТС 029/2012", clause="1")
    assert rank_of(gold, [other, other, gold], INDEX) == 3
    assert rank_of(gold, [gold], INDEX) == 1
    assert rank_of(gold, [other, other], INDEX) is None


# --- Негативные кейсы: где зачёт ставить НЕЛЬЗЯ ---

def test_same_clause_number_other_regulation():
    """п.33 в 040 (глазурь рыбы) и п.33 в 034 (мясо) — номер совпал,
    регламент другой: НЕ попадание."""
    fish = find_id(regulation_id="ТР ЕАЭС 040/2016", clause="33", appendix=None)
    meat = find_id(regulation_id="ТР ТС 034/2013", clause="33", appendix=None)
    assert not is_hit(fish, meat, INDEX)
    assert not is_hit(meat, fish, INDEX)


def test_same_clause_number_other_subsection():
    """022: п.1 подраздела 4.7 (срок годности) и п.1 подраздела 4.8
    (изготовитель) — НЕ попадание."""
    ids = {}
    for c in CHUNKS:
        if (c["regulation_id"] == "ТР ТС 022/2011" and c.get("clause") == "1"
                and (c.get("subsection") or "").split(".")[:2] in (["4", "7"], ["4", "8"])):
            ids[c["subsection"].split(" ")[0]] = c["chunk_id"]
    assert not is_hit(ids["4.7."], ids["4.8."], INDEX)


def test_appendix_numbering_restart():
    """005 прил.1: несколько разных «пунктов 1» (нумерация перезапускается
    в каждой внутренней таблице) — друг для друга НЕ попадание."""
    ones = [c["chunk_id"] for c in CHUNKS
            if c["regulation_id"] == "ТР ТС 005/2011"
            and c.get("appendix") == "1" and c.get("clause") == "1"]
    assert len(ones) > 1, "кейс перезапуска нумерации исчез из корпуса"
    assert not is_hit(ones[0], ones[1], INDEX)


def test_window_neighbor_not_hit():
    """Соседние окна одной таблицы приложения — разные чанки, НЕ попадание
    (консервативное правило, см. EVALUATION.md)."""
    windows = [c["chunk_id"] for c in CHUNKS
               if c["regulation_id"] == "ТР ТС 029/2012"
               and c.get("appendix") == "2" and c.get("is_table")]
    assert not is_hit(windows[0], windows[1], INDEX)


def test_body_clause_never_matches_appendix():
    """Пункт тела и чанк приложения не совпадают, даже если номера похожи."""
    body = find_id(regulation_id="ТР ТС 005/2011", clause="1", appendix=None)
    app = find_id(regulation_id="ТР ТС 005/2011", clause="1", appendix="1")
    assert not is_hit(body, app, INDEX)


def test_unknown_chunk_id_raises():
    """Неизвестный chunk_id — что из поиска, что из эталона — ошибка
    (рассинхронизация), а не молчаливый промах."""
    good = CHUNKS[0]["chunk_id"]
    for gold, retrieved in ((good, "tr_ts_005_2011:9999"),
                            ("tr_ts_005_2011:9999", good)):
        try:
            is_hit(gold, retrieved, INDEX)
            raise AssertionError("неизвестный chunk_id не вызвал ошибку")
        except ValueError:
            pass


# --- Схема и защита от рассинхронизации ---

def test_every_chunk_has_category():
    """Каждый чанк корпуса попадает ровно в одну из четырёх категорий."""
    for c in CHUNKS:
        assert category_of(c) in CATEGORIES, c["chunk_id"]


def test_validate_record():
    """Валидатор записей эталона ловит битые записи и пропускает корректные."""
    good_chunk = next(c for c in CHUNKS if category_of(c) == "body_clause")
    good = {"question": "Ê", "chunk_id": good_chunk["chunk_id"],
            "regulation_id": good_chunk["regulation_id"],
            "category": "body_clause"}
    assert validate_record(good, INDEX) == []
    assert validate_record({**good, "question": ""}, INDEX)
    assert validate_record({**good, "chunk_id": "нет:0000"}, INDEX)
    assert validate_record({**good, "category": "table_window"}, INDEX)
    assert validate_record({**good, "regulation_id": "ТР ТС 029/2012"}, INDEX)


def test_corpus_hash_guard():
    """Эталон от другой версии корпуса не загружается: chunk_id позиционные,
    после перепарсинга старые номера недействительны."""
    rec = {"question": "тест", "chunk_id": CHUNKS[0]["chunk_id"],
           "regulation_id": CHUNKS[0]["regulation_id"],
           "category": category_of(CHUNKS[0])}
    with tempfile.TemporaryDirectory() as tmp:
        gt = Path(tmp) / "gt.jsonl"
        gt.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
        meta = Path(tmp) / "meta.json"

        # Неправильный хэш → обязана быть ошибка.
        meta.write_text(json.dumps({"corpus_sha256": "0" * 64}), encoding="utf-8")
        try:
            load_ground_truth(gt, meta, CHUNKS_PATH)
            raise AssertionError("рассинхронизация корпуса не вызвала ошибку")
        except ValueError:
            pass

        # Правильный хэш → загрузка проходит.
        meta.write_text(json.dumps({"corpus_sha256": sha256_of(CHUNKS_PATH)}),
                        encoding="utf-8")
        records, index = load_ground_truth(gt, meta, CHUNKS_PATH)
        assert len(records) == 1 and records[0]["question"] == "тест"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"✅ {name}: {fn.__doc__ or ''}".strip())
        except Exception as e:  # и AssertionError, и сбой самого матчера
            failed += 1
            print(f"❌ {name}: {e or fn.__doc__}")
    print(f"\n{len(tests) - failed}/{len(tests)} проверок пройдено")
    sys.exit(1 if failed else 0)
