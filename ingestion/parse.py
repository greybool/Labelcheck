"""PDF регламентов → чанки с метаданными.

Читает data/sources.yaml (что парсить) и ingestion/cleanup.yaml (как чистить
и резать), пишет data/chunks.jsonl: один чанк = одна строка JSON.

Запуск из корня репозитория:  python ingestion/parse.py [--audit]
С флагом --audit дополнительно пишет data/dropped_lines.txt — все удалённые
строки для ручной проверки, что не отрезано ничего нужного.
"""

import json
import re
import sys
from pathlib import Path

import pdfplumber
import yaml

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"

# Заголовки разделов по типу структуры документа.
SECTION_PATTERNS = {
    "articles_upper": re.compile(r"^СТАТЬЯ\s+\d+\.?\s*.*$"),
    "articles_mixed": re.compile(r"^Статья\s+\d+\.?\s*.*$"),
    "roman_sections": re.compile(r"^[IVX]+\.\s+.+$"),
}
# Начало пункта: номер (возможно многоуровневый) с точкой, дальше не цифра.
# (?!\d) отсекает табличные коды вида «05.014».
CLAUSE_RE = re.compile(r"^(\d+(?:\.\d+)*)\.(?!\d)\s*(.*)$")
# Заголовок приложения — отдельной строкой: «Приложение 2» или «Приложение N 2».
APPENDIX_RE = re.compile(r"^Приложение\s*(?:N\s*)?(\d+)\s*$")


def pdf_to_lines(path: Path) -> list[str]:
    """Извлекает текст PDF постранично, возвращает список строк."""
    lines = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            lines.extend(text.split("\n"))
            page.flush_cache()
            page.get_textmap.cache_clear()
    return lines


def clean_lines(lines: list[str], rules: dict) -> tuple[list[str], list[str]]:
    """Удаляет служебные строки по правилам источника. Возвращает (оставленные, удалённые)."""
    contains = rules.get("drop_contains", [])
    regexes = [re.compile(r) for r in rules.get("drop_regex", [])]
    blocks = rules.get("drop_blocks", [])

    kept, dropped = [], []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Блок из нескольких строк (например, аннотация ГАРАНТа об изменениях).
        block_len = 0
        for b in blocks:
            if re.search(b["start"], line):
                for j in range(i, min(i + b["max_span"], len(lines))):
                    if re.search(b["end"], lines[j].strip()):
                        block_len = j - i + 1
                        break
        if block_len:
            dropped.extend(lines[i:i + block_len])
            i += block_len
            continue

        if any(s in line for s in contains) or any(r.match(line) for r in regexes):
            dropped.append(line)
        else:
            kept.append(line)
        i += 1
    return kept, dropped


def split_appendices(lines: list[str]) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """Отделяет тело документа от приложений по строкам-заголовкам «Приложение N»."""
    boundaries = [i for i, ln in enumerate(lines) if APPENDIX_RE.match(ln)]
    if not boundaries:
        return lines, []
    body = lines[: boundaries[0]]
    appendices = []
    for k, start in enumerate(boundaries):
        end = boundaries[k + 1] if k + 1 < len(boundaries) else len(lines)
        num = APPENDIX_RE.match(lines[start]).group(1)
        appendices.append((num, lines[start:end]))
    return body, appendices


def flush(chunks: list, buf: list[str], meta: dict, cfg: dict) -> None:
    """Закрывает накопленный чанк: короткие «заголовки» приклеивает к контексту, длинные режет."""
    text = "\n".join(buf).strip()
    if not text:
        return
    max_chars = cfg["max_chars"]
    if len(text) <= max_chars:
        chunks.append({**meta, "text": text})
        return
    # Длинный пункт: режем по строкам на части одинаковых метаданных.
    part, current = 1, []
    for ln in text.split("\n"):
        if sum(len(s) + 1 for s in current) + len(ln) > max_chars and current:
            chunks.append({**meta, "part": part, "text": "\n".join(current)})
            part += 1
            current = []
        current.append(ln)
    if current:
        chunks.append({**meta, "part": part, "text": "\n".join(current)})


def chunk_body(lines: list[str], structure: str, base: dict, cfg: dict) -> list[dict]:
    """Тело документа: раздел → пункты. Пункт = чанк."""
    section_re = SECTION_PATTERNS[structure]
    # До первого заголовка раздела — преамбула (титул, оглавление): выбрасываем.
    start = next((i for i, ln in enumerate(lines) if section_re.match(ln)), 0)

    chunks: list[dict] = []
    section, subsection, clause = None, None, None
    buf: list[str] = []

    def meta() -> dict:
        return {**base, "section": section, "subsection": subsection,
                "clause": clause, "appendix": None, "is_table": False}

    body = lines[start:]
    for i, ln in enumerate(body):
        if section_re.match(ln):
            flush(chunks, buf, meta(), cfg)
            buf, section, subsection, clause = [], ln.strip(), None, None
            continue
        m = CLAUSE_RE.match(ln)
        if m:
            flush(chunks, buf, meta(), cfg)
            number, rest = m.group(1), m.group(2)
            # Заголовок подраздела (например «4.1. Требования к …» в 022):
            # многоуровневый номер, короткий текст с Заглавной буквы, без точки
            # в конце, а следующая строка начинает пункт «1.» — иначе это обычный
            # пункт вида 11.1 с переносом строки.
            nxt = body[i + 1].strip() if i + 1 < len(body) else ""
            looks_like_heading = ("." in number and rest[:1].isupper()
                                  and len(rest) < cfg["min_chars"]
                                  and not rest.endswith((".", ";", ":"))
                                  and CLAUSE_RE.match(nxt))
            if looks_like_heading:
                subsection, clause, buf = f"{number}. {rest}".strip(), None, []
            else:
                clause, buf = number, [ln.strip()]
            continue
        buf.append(ln.strip())
    flush(chunks, buf, meta(), cfg)
    return chunks


def chunk_appendix(num: str, lines: list[str], base: dict, cfg: dict) -> list[dict]:
    """Приложение: с нумерацией пунктов — как тело; без неё (таблицы) — окнами строк."""
    clause_lines = sum(1 for ln in lines if CLAUSE_RE.match(ln))
    meta = {**base, "section": None, "subsection": None, "appendix": num}

    if clause_lines >= 3:
        chunks: list[dict] = []
        clause, buf = None, []
        for ln in lines:
            m = CLAUSE_RE.match(ln)
            if m:
                flush(chunks, buf, {**meta, "clause": clause, "is_table": False}, cfg)
                clause, buf = m.group(1), [ln.strip()]
            else:
                buf.append(ln.strip())
        flush(chunks, buf, {**meta, "clause": clause, "is_table": False}, cfg)
        # Если большинство «пунктов» — короткие обрывки, это таблица с
        # нумерованными строками (нормативы 021 и т.п.), а не текст с пунктами:
        # нарезка по пунктам тут вредна, переходим на окна.
        tiny = sum(1 for c in chunks if len(c["text"]) < cfg["min_chars"])
        if tiny <= len(chunks) * cfg["table_scrap_share"]:
            return chunks

    # Табличное приложение: окна по window_lines строк с перехлёстом,
    # шапка приложения повторяется в каждом чанке.
    header = lines[: cfg["header_lines"]]
    body = lines[cfg["header_lines"]:]
    step = max(1, cfg["window_lines"] - cfg["window_overlap"])
    chunks = []
    for i in range(0, len(body), step):
        window = body[i:i + cfg["window_lines"]]
        new_part = window if i == 0 else window[cfg["window_overlap"]:]
        if not new_part:  # хвост целиком повторяет предыдущее окно
            break
        if len("\n".join(new_part).strip()) < cfg["min_chars"] and chunks:
            chunks[-1]["text"] += "\n" + "\n".join(new_part).strip()  # хвост — к предыдущему
        else:
            chunks.append({**meta, "clause": None, "is_table": True,
                           "text": "\n".join(header + window).strip()})
    return chunks


def main() -> int:
    audit = "--audit" in sys.argv
    sources = yaml.safe_load(open(DATA / "sources.yaml", encoding="utf-8"))["documents"]
    config = yaml.safe_load(open(Path(__file__).parent / "cleanup.yaml", encoding="utf-8"))
    cfg = config["chunking"]

    all_chunks, all_dropped = [], []
    for doc in sources:
        pdf_path = DATA / doc["file"]
        stem = pdf_path.stem
        structure = config["structures"][stem]
        rules = config["cleanup"][doc["text_source"]]

        raw_lines = pdf_to_lines(pdf_path)
        lines, dropped = clean_lines(raw_lines, rules)
        body, appendices = split_appendices(lines)

        base = {"regulation_id": doc["id"], "regulation_name": doc["title"],
                "edition": doc["edition"]}
        doc_chunks = chunk_body(body, structure, base, cfg)
        for num, app_lines in appendices:
            doc_chunks.extend(chunk_appendix(num, app_lines, base, cfg))

        # Пост-фильтр: крошечный чанк тела без номера пункта — это обрывок
        # (перенос заглавного заголовка, осиротевшее примечание «в редакции…»),
        # а не норма. Нормативный текст без номера (определения) всегда длиннее.
        # Каждый выброшенный обрывок попадает в аудит с пометкой [SCRAP] —
        # потеря текста не бывает бесшумной.
        kept_chunks = []
        for c in doc_chunks:
            if (c.get("clause") or c.get("appendix")
                    or len(c["text"]) >= cfg["min_chars"]):
                kept_chunks.append(c)
            else:
                all_dropped.append(f"[SCRAP {stem}] {c['text']}")
        scraps = len(doc_chunks) - len(kept_chunks)
        doc_chunks = kept_chunks

        for i, ch in enumerate(doc_chunks):
            ch["chunk_id"] = f"{stem}:{i:04d}"
        all_chunks.extend(doc_chunks)
        all_dropped.extend(f"[{stem}] {ln}" for ln in dropped)

        with_clause = sum(1 for c in doc_chunks if c.get("clause"))
        in_app = sum(1 for c in doc_chunks if c.get("appendix"))
        print(f"{doc['id']:<18} чанков: {len(doc_chunks):>4} "
              f"(с пунктом: {with_clause}, из приложений: {in_app}, "
              f"удалено строк: {len(dropped)}, обрывков: {scraps})")

    out = DATA / "chunks.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for ch in all_chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    print(f"\nИтого {len(all_chunks)} чанков → {out}")

    if audit:
        audit_path = DATA / "dropped_lines.txt"
        audit_path.write_text("\n".join(all_dropped), encoding="utf-8")
        print(f"Удалённые строки ({len(all_dropped)}) → {audit_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
