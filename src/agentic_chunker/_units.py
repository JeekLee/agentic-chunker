"""Build source-preserving evidence units before agentic grouping."""
from __future__ import annotations

from ._models import Block, Chunk
from ._tables import split_structured_blocks


def build_units(blocks: list[Block]) -> list[Chunk]:
    """Convert split blocks into source-preserving evidence units.

    Unit ``text`` is always displayable source text. LLM grouping may later
    combine units, but must not rewrite their text.
    """
    text_blocks, structured = split_structured_blocks(blocks)
    text_units = [_text_unit(block) for block in text_blocks if not _is_placeholder(block.text)]
    ordered = _order_units([*text_units, *structured])
    return _order_units(_coalesce_heading_units(ordered))


def _text_unit(block: Block) -> Chunk:
    title = block.text.splitlines()[0][:80] if block.text else ""
    kind = "heading" if _is_text_heading(block.text) else "text"
    return Chunk(
        index=0,
        text=block.text,
        title=title,
        summary=block.text[:160],
        source_spans=[(block.char_start, block.char_end)],
        embedding_text=block.text,
        metadata={
            "common": {
                "chunk_kind": kind,
                "section_path": [block.header] if block.header else [],
                "display_format": "plain",
            },
        },
    )


def _order_units(units: list[Chunk]) -> list[Chunk]:
    def start(unit: Chunk) -> int:
        return min((s[0] for s in unit.source_spans), default=0)

    ordered = sorted(units, key=start)
    for i, unit in enumerate(ordered):
        unit.index = i
        unit.metadata.setdefault("common", {
            "chunk_kind": "text",
            "section_path": [],
            "display_format": "plain",
        })
        if not unit.embedding_text:
            unit.embedding_text = "\n".join(p for p in (unit.title, unit.summary, unit.text) if p)
    return ordered


def _is_placeholder(text: str) -> bool:
    normalized = text.strip()
    return normalized in {"...", "…", "⋮", "⋯"}


def _is_text_heading(text: str) -> bool:
    stripped = text.strip()
    return len(stripped) <= 80 and stripped.startswith(("□", "■", "▶"))


def _coalesce_heading_units(units: list[Chunk]) -> list[Chunk]:
    coalesced: list[Chunk] = []
    pending_headings: list[Chunk] = []
    for unit in units:
        if _unit_kind(unit) == "heading":
            pending_headings.append(unit)
            continue
        if pending_headings:
            coalesced.append(_prepend_headings(pending_headings, unit))
            pending_headings = []
        else:
            coalesced.append(unit)
    coalesced.extend(pending_headings)
    return coalesced


def _prepend_headings(headings: list[Chunk], unit: Chunk) -> Chunk:
    heading_texts = [heading.text for heading in headings if heading.text]
    metadata = _copy_metadata(unit.metadata)
    common = dict(metadata.get("common", {}))
    section_path = list(common.get("section_path", []))
    for heading in heading_texts:
        if heading not in section_path:
            section_path.append(heading)
    common["section_path"] = section_path
    common["layout_headings"] = heading_texts
    metadata["common"] = common

    return Chunk(
        index=unit.index,
        text="\n\n".join([*heading_texts, unit.text]),
        title=(heading_texts[0] if heading_texts else unit.title)[:80],
        summary=unit.summary,
        keywords=_merge_keywords([*headings, unit]),
        source_spans=[span for heading in headings for span in heading.source_spans] + list(unit.source_spans),
        embedding_text="\n".join([*heading_texts, unit.embedding_text or unit.text]),
        metadata=metadata,
    )


def _unit_kind(unit: Chunk) -> str:
    return unit.metadata.get("common", {}).get("chunk_kind", "text")


def _merge_keywords(units: list[Chunk]) -> list[str]:
    keywords: list[str] = []
    for unit in units:
        for keyword in unit.keywords:
            if keyword and keyword not in keywords:
                keywords.append(keyword)
    return keywords[:12]


def _copy_metadata(metadata: dict) -> dict:
    copied = {}
    for key, value in metadata.items():
        if isinstance(value, dict):
            copied[key] = dict(value)
        elif isinstance(value, list):
            copied[key] = list(value)
        else:
            copied[key] = value
    return copied
