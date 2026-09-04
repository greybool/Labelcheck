"""Гигиена ПУБЛИЧНОГО репозитория: в индексе git не должно быть секретов и
личных данных компании (правило 7 CLAUDE.md).

Появился после инцидента 04.09: архив `_to_delete/stage_tmp.tgz` с живым
.env (ключ OpenAI) и текстами реальных макетов четыре дня лежал в публичном
репозитории — grep по текстам такой бинарник не ловит. Тест смотрит на
индекс git (то, что реально уйдёт в коммит), а не на файлы на диске.

Запуск из корня репозитория:  python tests/test_repo_hygiene.py
(совместим и с pytest: pytest tests/). Вне git-репозитория (чистая
распаковка, контейнер без .git) проверки индекса пропускаются.
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Что НИКОГДА не должно быть в индексе: (регулярное выражение по пути, почему)
FORBIDDEN = [
    (r"(^|/)\.env$", "живой .env с ключом"),
    (r"(^|/)\.env\.(?!example$)", ".env.* кроме шаблона"),
    (r"\.(tgz|tar|tar\.gz|zip|7z|rar)$", "архивы — переносы между сессиями"),
    (r"^_to_delete/", "локальная корзина"),
    (r"^handoffs/", "handoff'ы: личные пути и имена макетов"),
    (r"^data/layouts", "layout'ы: тексты реальных макетов"),
    (r"^data/reports/", "отчёты: тексты и вердикты реальных макетов"),
    (r"^data/samples_private/", "PDF реальных макетов"),
    (r"^data/ui_uploads/", "загруженные через UI макеты"),
    (r"^data/vision_gt/", "vision-эталон: тексты макетов"),
    (r"^data/labelcheck\.db$", "рабочий журнал (демо-копия — отдельный файл)"),
    (r"^data/(verdict_cache|actions_cache)\.json$", "кэши с текстами макетов"),
    (r"^data/embeddings\.npz$", "кэш векторов — пересоздаётся"),
    (r"^data/demo_forbidden\.txt$", "личный стоп-список"),
    (r"^\.(agents|claude)/", "локальные скиллы инструментов"),
    (r"\.(pdf)$", "PDF вне корпуса регламентов"),
]
# Исключения из последнего правила: корпус регламентов и синтетический
# демо-макет (evaluation/make_demo_label.py) лежат в git.
ALLOWED_PDF = re.compile(r"^data/(raw/tr_(ts|eaeu)_\d{3}_\d{4}|samples/demo_[a-z_]+)\.pdf$")


def tracked_files() -> list[str] | None:
    """Пути из индекса git; None — не git-репозиторий или нет git."""
    if not (ROOT / ".git").exists():
        return None
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True,
                             capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return [p for p in out.decode("utf-8", "replace").split("\0") if p]


def offending(paths: list[str]) -> list[tuple[str, str]]:
    bad = []
    for p in paths:
        if ALLOWED_PDF.match(p):
            continue
        for pattern, why in FORBIDDEN:
            if re.search(pattern, p):
                bad.append((p, why))
                break
    return bad


def test_forbidden_patterns_catch_known_leaks():
    """Сам список правил: ловит инцидент 04.09 и типичные утечки, не
    трогает корпус регламентов и шаблон .env."""
    sample = ["_to_delete/stage_tmp.tgz", ".env", "data/layouts/mandu.json",
              "data/labelcheck.db", "handoffs/handoff-09.md", "backup.zip",
              "data/samples_private/x.pdf", ".env.local"]
    assert [p for p, _ in offending(sample)] == sample
    ok = ["data/raw/tr_ts_022_2011.pdf", "data/samples/demo_label.pdf",
          ".env.example", "data/labelcheck.demo.db",
          "data/chunks.jsonl", "labelcheck/app.py", "docs/REVIEW-LOG.md"]
    assert offending(ok) == []


def test_git_index_has_no_secrets_or_private_data():
    """Индекс git текущего репозитория чист (вне git — пропуск)."""
    paths = tracked_files()
    if paths is None:
        return
    bad = offending(paths)
    assert not bad, "в индексе git найдены запрещённые файлы:\n" + "\n".join(
        f"  {p}  — {why}" for p, why in bad)


def test_env_example_has_no_key():
    """Шаблон .env.example не содержит значения ключа."""
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    m = re.search(r"^OPENAI_API_KEY=(.*)$", text, re.M)
    assert m and m.group(1).strip() == "", "в .env.example вписан ключ"


def test_gitignore_covers_private_paths():
    """.gitignore закрывает всё из списка FORBIDDEN, что вообще может лежать
    в рабочей папке (иначе один неверный `git add .` — и утечка)."""
    gi = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for needle in (".env", "_to_delete/", "handoffs/", "data/layouts/",
                   "data/reports/", "samples_private/", "data/ui_uploads/",
                   "data/vision_gt/", "data/labelcheck.db",
                   "data/verdict_cache.json", "data/embeddings.npz"):
        assert needle in gi, f"в .gitignore нет {needle}"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"✅ {name}: {fn.__doc__ or ''}".strip())
        except Exception as e:
            failed += 1
            print(f"❌ {name}: {e or fn.__doc__}")
    print(f"\n{len(tests) - failed}/{len(tests)} проверок пройдено")
    sys.exit(1 if failed else 0)
