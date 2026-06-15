"""Build source-preserving evidence units before agentic grouping."""
from __future__ import annotations

from ._common import Block, Chunk
from ._tables import link_table_references, split_structured_blocks


def build_units(blocks: list[Block]) -> list[Chunk]:
    """Convert split blocks into source-preserving evidence units.

    Unit ``text`` is always displayable source text. LLM grouping may later
    combine units, but must not rewrite their text.
    """
    text_blocks, structured = split_structured_blocks(blocks)
    text_units = [_text_unit(block) for block in text_blocks]
    units = _order_units([*text_units, *structured])
    link_table_references(units)
    return units


def _text_unit(block: Block) -> Chunk:
    title = block.text.splitlines()[0][:80] if block.text else ""
    return Chunk(
        index=0,
        text=block.text,
        title=title,
        summary=block.text[:160],
        source_spans=[(block.char_start, block.char_end)],
        embedding_text=block.text,
        metadata={
            "common": {
                "chunk_kind": "text",
                "section_path": [block.header] if block.header else [],
                "display_format": "plain",
            },
            "references": {"referenced_tables": [], "linked_table_indices": []},
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
        unit.metadata.setdefault("references", {"referenced_tables": [], "linked_table_indices": []})
        if not unit.embedding_text:
            unit.embedding_text = "\n".join(p for p in (unit.title, unit.summary, unit.text) if p)
    return ordered
