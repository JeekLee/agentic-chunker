"""Shared dataclasses for agentic-chunker."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


DocumentNodeType = Literal[
    "document",
    "section",
    "chunk",
    "table",
    "figure",
    "code_block",
]

DocumentEdgeType = Literal[
    "HAS_SECTION",
    "HAS_CHILD",
    "HAS_CHUNK",
    "NEXT",
    "PREVIOUS",
    "REFERS_TO",
    "DESCRIBES",
    "HAS_TABLE",
    "HAS_FIGURE",
    "HAS_CAPTION",
    "CONTINUES",
]


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
    source_text: str = ""


@dataclass
class DocumentNode:
    id: str
    type: DocumentNodeType
    text: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class DocumentEdge:
    source_id: str
    target_id: str
    type: DocumentEdgeType
    metadata: dict = field(default_factory=dict)


@dataclass
class DocumentGraph:
    nodes: list[DocumentNode] = field(default_factory=list)
    edges: list[DocumentEdge] = field(default_factory=list)


@dataclass(init=False)
class Chunk:
    """A semantically coherent chunk emitted by the chunker.

    Dataclass serialization intentionally exposes the public RAG payload:
    ``source``, ``summary``, ``keywords``, ``questions_answered``, and
    ``document_graph``. Compatibility attributes such as ``text`` and
    ``metadata`` remain available for internal pipeline code and existing
    callers, but they are not part of the default serialized payload.
    """

    source: str
    summary: str
    keywords: list[str]
    questions_answered: list[str]
    document_graph: DocumentGraph

    def __init__(
        self,
        index: int = 0,
        source: str = "",
        *,
        text: str | None = None,
        summary: str = "",
        keywords: list[str] | None = None,
        questions_answered: list[str] | None = None,
        document_graph: DocumentGraph | None = None,
        title: str = "",
        source_spans: list[tuple[int, int]] | None = None,
        embedding_text: str = "",
        metadata: dict | None = None,
        id: str | None = None,
    ) -> None:
        self.index = index
        self.id = id or f"chunk:{index}"
        self.source = source if source else (text or "")
        self.summary = summary
        self.keywords = list(keywords or [])
        self.questions_answered = list(questions_answered or [])
        self.document_graph = document_graph or DocumentGraph()
        self.title = title
        self.source_spans = list(source_spans or [])
        self.embedding_text = embedding_text
        self.metadata = dict(metadata or {})

    @property
    def text(self) -> str:
        return self.source

    @text.setter
    def text(self, value: str) -> None:
        self.source = value
