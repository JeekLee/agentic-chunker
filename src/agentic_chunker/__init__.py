"""agentic-chunker — source-preserving agentic chunking for RAG.

    from agentic_chunker import AgenticChunker, LlmConfig, Chunk

    chunker = AgenticChunker(
        llm=LlmConfig(url="http://localhost:10080/v1", api_key="...", model="qwen3-..."),
    )
    chunks = chunker.chunk(markdown_text)
"""
from __future__ import annotations

from ._common import (
    Block,
    Chunk,
    DocumentEdge,
    DocumentGraph,
    DocumentNode,
    Proposition,
)
from ._document_graph import attach_document_graphs as _attach_document_graphs
from ._domain import (
    DocumentContext,
    DomainExtractionResult,
    DomainExtractor,
    DomainSchema,
    Entity,
    Triple,
    run_domain_extraction as _run_domain_extraction,
)
from ._split import split as _split
from ._unit_agent import group_units as _group_units
from ._units import build_units as _build_units
from .llm import LlmConfig


class AgenticChunker:
    """Markdown/text → source-preserving chunks via evidence-unit agentic grouping.

    Args:
        llm: LlmConfig for an OpenAI-compatible endpoint.
        max_propositions_per_chunk: compatibility name; used as max evidence units per chunk.
        window_size: evidence units per grouping call; larger documents are windowed.
        min_extract_chars: retained for API compatibility; no longer used by the unit grouper.
        max_concurrency: thread cap for parallel window grouping.
        document_graph: attach per-chunk DocumentGraph neighborhoods when true.
        domain_schema: optional schema for LLM-based domain extraction.
        domain_extractor: optional custom domain extractor. Mutually exclusive with domain_schema.
    """

    def __init__(
        self,
        *,
        llm: LlmConfig,
        max_propositions_per_chunk: int = 10,
        window_size: int = 40,
        min_extract_chars: int = 20,
        max_concurrency: int = 4,
        document_graph: bool = True,
        domain_schema: DomainSchema | None = None,
        domain_extractor: DomainExtractor | None = None,
    ) -> None:
        if domain_schema is not None and domain_extractor is not None:
            raise ValueError("domain_schema and domain_extractor are mutually exclusive")
        self._llm = llm
        self._max_props = max_propositions_per_chunk
        self._window_size = window_size
        self._min_extract_chars = min_extract_chars
        self._concurrency = max_concurrency
        self._document_graph = document_graph
        self._domain_schema = domain_schema
        self._domain_extractor = domain_extractor
        self.domain_extraction = DomainExtractionResult()

    def chunk(self, markdown: str) -> list[Chunk]:
        blocks = _split(markdown)
        if not blocks:
            return []
        units = _build_units(blocks)
        chunks = _group_units(
            units,
            self._llm,
            max_units=self._max_props,
            window_size=self._window_size,
            concurrency=self._concurrency,
        )
        if self._document_graph:
            _attach_document_graphs(chunks)
        if self._domain_schema is not None or self._domain_extractor is not None:
            self.domain_extraction = _run_domain_extraction(
                chunks,
                self._llm,
                schema=self._domain_schema,
                extractor=self._domain_extractor,
                concurrency=self._concurrency,
            )
        return chunks


__all__ = [
    "AgenticChunker",
    "LlmConfig",
    "Chunk",
    "Block",
    "Proposition",
    "DocumentNode",
    "DocumentEdge",
    "DocumentGraph",
    "Entity",
    "Triple",
    "DomainExtractionResult",
    "DomainSchema",
    "DocumentContext",
    "DomainExtractor",
]
