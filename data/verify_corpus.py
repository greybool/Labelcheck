"""Проверка целостности корпуса регламентов.

Сверяет файлы в data/raw/ с описанием в data/sources.yaml:
файл существует, размер совпадает, SHA256-хэш совпадает.

Запуск из корня репозитория:  python data/verify_corpus.py
Код выхода: 0 — корпус в порядке, 1 — есть проблемы.
"""

import hashlib
import sys
from pathlib import Path

import yaml

# Папка, где лежит этот скрипт (data/), — все пути в sources.yaml
# отсчитываются от неё, поэтому скрипт работает из любой директории.
DATA_DIR = Path(__file__).parent


def sha256_of(path: Path) -> str:
    """Считает SHA256-хэш файла, читая его кусками по 1 МБ."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def check_document(doc: dict) -> tuple[bool, str]:
    """Проверяет один документ. Возвращает (успех, текст статуса)."""
    path = DATA_DIR / doc["file"]

    if not path.exists():
        return False, "ФАЙЛ НЕ НАЙДЕН"

    size = path.stat().st_size
    if size != doc["size_bytes"]:
        return False, f"РАЗМЕР {size}, ожидался {doc['size_bytes']}"

    if sha256_of(path) != doc["sha256"]:
        return False, "SHA256 НЕ СОВПАДАЕТ (файл изменён или повреждён)"

    return True, "OK"


def main() -> int:
    sources = DATA_DIR / "sources.yaml"
    if not sources.exists():
        print(f"Не найден {sources}")
        return 1

    with open(sources, encoding="utf-8") as f:
        documents = yaml.safe_load(f)["documents"]

    print(f"Проверка корпуса: {len(documents)} документов\n")
    failures = 0
    for doc in documents:
        ok, status = check_document(doc)
        mark = "✅" if ok else "❌"
        if not ok:
            failures += 1
        print(f"{mark} {doc['id']:<18} {doc['edition']:<55} {status}")

    print()
    if failures:
        print(f"Проблем: {failures}. Сверь файлы в data/raw/ с data/sources.yaml.")
        return 1
    print("Корпус в порядке.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
