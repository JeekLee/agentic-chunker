"""Deterministic table-aware chunking helpers.

Tables are evidence surfaces in RAG: they should stay displayable as Markdown
while carrying a flattened embedding string for retrieval.
"""
from __future__ import annotations

import re

from ._common import Block, Chunk

_CAPTION_RE = re.compile(r"^\**\s*\[?\s*표\s*(\d+)\s*\]?\s*\**$")
_TABLE_REF_RE = re.compile(r"(?:\[?\s*표\s*(\d+)\s*\]?|→\s*표\s*(\d+))")
_BR_RE = re.compile(r"\s*<br\s*/?>\s*", re.IGNORECASE)


def split_structured_blocks(blocks: list[Block]) -> tuple[list[Block], list[Chunk]]:
    """Return ``(text_blocks, structured_chunks)``.

    Markdown tables and standalone ``[표 N]`` captions are removed from the LLM
    proposition path. Structured chunks keep Markdown table text for display and
    put retrieval-oriented text in ``embedding_text``.
    """
    text_blocks: list[Block] = []
    structured: list[Chunk] = []
    pending_table_id = ""
    pending_caption_block: Block | None = None

    for block in blocks:
        caption = _caption_id(block.text)
        if caption:
            pending_table_id = caption
            pending_caption_block = block
            continue

        table = _parse_table(block.text)
        if table is None:
            if pending_caption_block is not None:
                structured.append(_caption_chunk(pending_caption_block, pending_table_id))
            text_blocks.append(block)
            pending_table_id = ""
            pending_caption_block = None
            continue

        chunks = _chunks_for_table(block, table, pending_table_id, pending_caption_block)
        structured.extend(chunks)
        pending_table_id = ""
        pending_caption_block = None

    if pending_table_id and pending_caption_block is not None:
        structured.append(_caption_chunk(pending_caption_block, pending_table_id))

    return text_blocks, structured


def link_table_references(chunks: list[Chunk]) -> None:
    """Populate table reference metadata for every chunk kind.

    References are extracted from ``chunk.text`` only because that is the
    displayable source/evidence text. LLM-generated embedding text is not used
    for reference discovery.
    """
    table_indices: dict[str, list[int]] = {}
    for chunk in chunks:
        table_meta = chunk.metadata.get("table", {})
        table_id = table_meta.get("table_id")
        if table_id:
            table_indices.setdefault(table_id, []).append(chunk.index)

    backrefs: dict[int, list[int]] = {}
    for chunk in chunks:
        refs_meta = chunk.metadata.setdefault("references", {})
        existing = refs_meta.get("referenced_tables", [])
        refs: list[str] = [r for r in existing if isinstance(r, str)]
        own_table_id = chunk.metadata.get("table", {}).get("table_id")
        for ref in _table_refs(chunk.text):
            if ref == own_table_id:
                continue
            if ref not in refs:
                refs.append(ref)
        refs_meta["referenced_tables"] = refs

        linked: list[int] = []
        for table_id in refs:
            for idx in table_indices.get(table_id, []):
                if idx == chunk.index or table_id == own_table_id:
                    continue
                if idx not in linked:
                    linked.append(idx)
                    backrefs.setdefault(idx, []).append(chunk.index)
        refs_meta["linked_table_indices"] = linked

    for chunk in chunks:
        refs_meta = chunk.metadata.setdefault("references", {})
        refs_meta["referenced_by_indices"] = sorted(set(backrefs.get(chunk.index, [])))


def _caption_id(text: str) -> str:
    m = _CAPTION_RE.match(text.strip())
    return f"표 {m.group(1)}" if m else ""


def _caption_chunk(block: Block, table_id: str) -> Chunk:
    return Chunk(
        index=0,
        text=block.text,
        title=table_id,
        summary=table_id,
        source_spans=[(block.char_start, block.char_end)],
        embedding_text=table_id,
        metadata={
            "common": {"chunk_kind": "table_caption", "section_path": _section_path(block), "display_format": "markdown"},
            "table": {"table_id": table_id},
            "references": {"referenced_tables": [], "linked_table_indices": []},
        },
    )


def _table_refs(text: str) -> list[str]:
    refs: list[str] = []
    for m in _TABLE_REF_RE.finditer(text):
        num = m.group(1) or m.group(2)
        ref = f"표 {num}"
        if ref not in refs:
            refs.append(ref)
    return refs


def _parse_table(text: str) -> tuple[list[str], list[list[str]]] | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2 or not all(line.startswith("|") for line in lines[:2]):
        return None

    header = _parse_row(lines[0])
    separator = _parse_row(lines[1])
    if not header or not _is_separator(separator):
        return None

    rows = [_parse_row(line) for line in lines[2:] if line.startswith("|")]
    rows = [row for row in rows if row]
    return header, rows


def _parse_row(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")]


def _is_separator(row: list[str]) -> bool:
    if not row:
        return False
    return all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in row)


def _chunks_for_table(
    block: Block,
    table: tuple[list[str], list[list[str]]],
    table_id: str,
    caption_block: Block | None,
) -> list[Chunk]:
    headers, rows = table
    return _general_table_chunks(block, headers, rows, table_id, caption_block)


def _general_table_chunks(
    block: Block,
    headers: list[str],
    rows: list[list[str]],
    table_id: str,
    caption_block: Block | None,
) -> list[Chunk]:
    max_rows = 12
    parts = [rows[i:i + max_rows] for i in range(0, len(rows), max_rows)] or [[]]
    total = len(parts)
    chunks: list[Chunk] = []
    for part_index, part_rows in enumerate(parts, start=1):
        padded = [_pad(row, len(headers)) for row in part_rows]
        text = _table_display_text(caption_block, _format_table(headers, padded))
        title_base = table_id or _clean_cell(" ".join(headers))[:80] or "table"
        title = title_base if total == 1 else f"{title_base} ({part_index}/{total})"
        clean_headers = [_clean_cell(h) for h in headers]
        embedding = _table_embedding(table_id, clean_headers, padded, part_index, total)
        summary = _table_summary(title_base, clean_headers, len(rows), part_index, total)
        keywords = _table_keywords(table_id, clean_headers, title_base)
        chunks.append(Chunk(
            index=0,
            text=text,
            title=title,
            summary=summary,
            keywords=keywords,
            source_spans=_spans(block, caption_block),
            embedding_text=embedding,
            metadata={
                "common": {"chunk_kind": "table" if total == 1 else "table_part", "section_path": _section_path(block), "display_format": "markdown_table"},
                "table": {
                    "table_id": table_id,
                    "headers": clean_headers,
                    "row_count": len(rows),
                    "part_index": part_index if total > 1 else None,
                    "part_total": total if total > 1 else None,
                },
                "references": {"referenced_tables": [], "linked_table_indices": []},
            },
        ))
    return chunks


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    width = len(headers)
    header_line = "| " + " | ".join(_pad(headers, width)) + " |"
    sep_line = "| " + " | ".join("---" for _ in range(width)) + " |"
    row_lines = ["| " + " | ".join(_pad(row, width)) + " |" for row in rows]
    return "\n".join([header_line, sep_line, *row_lines])


def _table_display_text(caption_block: Block | None, table_text: str) -> str:
    if caption_block is None or not caption_block.text:
        return table_text
    return f"{caption_block.text}\n\n{table_text}"


def _table_embedding(
    table_id: str,
    headers: list[str],
    rows: list[list[str]],
    part_index: int,
    part_total: int,
) -> str:
    prefix = table_id or "표"
    if part_total > 1:
        prefix = f"{prefix} part {part_index}/{part_total}"
    lines = [f"{prefix}. 열: {', '.join(headers)}."]
    for row in rows:
        clean_cells = [_clean_cell(cell) for cell in _pad(row, len(headers))]
        pairs = [f"{headers[i]}={clean_cells[i]}" for i in range(len(headers)) if clean_cells[i]]
        if pairs:
            lines.append("; ".join(pairs) + ".")
    return " ".join(lines)


def _table_summary(
    title: str,
    headers: list[str],
    row_count: int,
    part_index: int,
    part_total: int,
) -> str:
    label = title or "표"
    header_text = ", ".join(h for h in headers if h)
    if part_total > 1:
        return f"{label} 표의 {row_count}개 행 중 {part_index}/{part_total} 부분이며, 열은 {header_text}입니다."
    return f"{label} 표이며, {row_count}개 행과 {header_text} 열을 포함합니다."


def _table_keywords(table_id: str, headers: list[str], title: str) -> list[str]:
    keywords: list[str] = []
    for item in [table_id, title, *headers]:
        cleaned = _clean_cell(item)
        if cleaned and cleaned not in keywords:
            keywords.append(cleaned)
    return keywords[:12]


def _clean_cell(text: str) -> str:
    text = _BR_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _pad(row: list[str], width: int) -> list[str]:
    if len(row) >= width:
        return row[:width]
    return [*row, *([""] * (width - len(row)))]


def _section_path(block: Block) -> list[str]:
    return [block.header] if block.header else []


def _spans(block: Block, caption_block: Block | None) -> list[tuple[int, int]]:
    spans = []
    if caption_block is not None:
        spans.append((caption_block.char_start, caption_block.char_end))
    spans.append((block.char_start, block.char_end))
    return spans
