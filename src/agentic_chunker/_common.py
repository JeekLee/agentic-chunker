"""Shared dataclasses for agentic-chunker."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Block:
    """A unit of source text from the structural pre-split."""
    text: str
    char_start: int
    char_end: int
    header: str | None = None


@dataclass
class Proposition:
    """An atomic, self-contained statement extracted from a Block."""
    text: str
    char_start: int
    char_end: int
    header: str | None = None


@dataclass
class Chunk:
    """A semantically coherent chunk emitted by the chunker."""
    index: int
    text: str
    title: str = ""
    summary: str = ""
    keywords: list[str] = field(default_factory=list)
    source_spans: list[tuple[int, int]] = field(default_factory=list)
