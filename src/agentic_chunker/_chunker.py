"""Public chunker implementation."""
from __future__ import annotations

from ._graph import attach_document_graphs as _attach_document_graphs
from ._domain import run_domain_extraction as _run_domain_extraction
from ._domain_models import (
    DomainExtractionResult,
    DomainExtractor,
    DomainSchema,
)
from ._models import Chunk, DocumentGraph
from ._result import ChunkingResult
from ._split import split as _split
from ._grouping import group_units as _group_units
from ._units import build_units as _build_units
from .llm import LlmConfig


class AgenticChunker:
    """Markdown/text to source-preserving chunks via evidence-unit agentic grouping.

    Args:
        llm: LlmConfig for an OpenAI-compatible endpoint.
        max_units_per_chunk: preferred name for the max evidence units per chunk.
        max_propositions_per_chunk: compatibility alias for max_units_per_chunk.
        window_size: evidence units per grouping call; larger documents are windowed.
        min_extract_chars: retained for API compatibility; no longer used by the unit grouper.
        max_concurrency: thread cap for parallel window grouping.
        document_graph: attach per-chunk DocumentGraph neighborhoods when true.
        domain_schema: optional schema for LLM-based domain extraction.
        domain_extractor: optional custom domain extractor. Mutually exclusive with domain_schema.
        entities: convenience domain entity types, used to build DomainSchema.
        relations: convenience domain relation types, used to build DomainSchema.
        extraction_models: user-defined structured extraction model classes.
        domain_instructions: additional LLM extraction instructions for convenience schema.
    """

    def __init__(
        self,
        *,
        llm: LlmConfig,
        max_units_per_chunk: int = 10,
        max_propositions_per_chunk: int | None = None,
        window_size: int = 40,
        min_extract_chars: int = 20,
        max_concurrency: int = 4,
        document_graph: bool = True,
        domain_schema: DomainSchema | None = None,
        domain_extractor: DomainExtractor | None = None,
        entities: list[str] | None = None,
        relations: list[str] | None = None,
        extraction_models: list[type] | None = None,
        domain_instructions: str = "",
    ) -> None:
        schema = _resolve_domain_schema(
            domain_schema=domain_schema,
            domain_extractor=domain_extractor,
            entities=entities,
            relations=relations,
            extraction_models=extraction_models,
            domain_instructions=domain_instructions,
        )
        self._llm = llm
        if (
            max_propositions_per_chunk is not None
            and max_units_per_chunk != 10
            and max_units_per_chunk != max_propositions_per_chunk
        ):
            raise ValueError(
                "max_units_per_chunk and max_propositions_per_chunk cannot both be set"
            )
        self._max_units = (
            max_propositions_per_chunk
            if max_propositions_per_chunk is not None
            else max_units_per_chunk
        )
        self._window_size = window_size
        self._min_extract_chars = min_extract_chars
        self._concurrency = max_concurrency
        self._document_graph = document_graph
        self._domain_schema = schema
        self._domain_extractor = domain_extractor
        self.document_graph = DocumentGraph()
        self.domain_extraction = DomainExtractionResult()
        self.result = ChunkingResult()

    def chunk(self, markdown: str) -> list[Chunk]:
        return self.chunk_document(markdown).chunks

    def chunk_document(self, markdown: str) -> ChunkingResult:
        blocks = _split(markdown)
        if not blocks:
            self.document_graph = DocumentGraph()
            self.domain_extraction = DomainExtractionResult()
            self.result = ChunkingResult()
            return self.result
        units = _build_units(blocks)
        chunks = _group_units(
            units,
            self._llm,
            max_units=self._max_units,
            window_size=self._window_size,
            concurrency=self._concurrency,
        )
        document_graph = DocumentGraph()
        if self._document_graph:
            document_graph = _attach_document_graphs(chunks)
        if self._domain_schema is not None or self._domain_extractor is not None:
            self.domain_extraction = _run_domain_extraction(
                chunks,
                self._llm,
                schema=self._domain_schema,
                extractor=self._domain_extractor,
                concurrency=self._concurrency,
            )
        else:
            self.domain_extraction = DomainExtractionResult()
        self.document_graph = document_graph
        self.result = ChunkingResult.from_domain_result(
            chunks=chunks,
            document_graph=document_graph,
            domain_extraction=self.domain_extraction,
        )
        return self.result


def _resolve_domain_schema(
    *,
    domain_schema: DomainSchema | None,
    domain_extractor: DomainExtractor | None,
    entities: list[str] | None,
    relations: list[str] | None,
    extraction_models: list[type] | None,
    domain_instructions: str,
) -> DomainSchema | None:
    convenience_used = any([
        entities is not None,
        relations is not None,
        extraction_models is not None,
        bool(domain_instructions),
    ])
    if domain_extractor is not None and (domain_schema is not None or convenience_used):
        raise ValueError("domain_extractor is mutually exclusive with schema-based extraction")
    if domain_schema is not None and convenience_used:
        raise ValueError(
            "domain_schema cannot be combined with entities, relations, "
            "extraction_models, or domain_instructions"
        )
    if domain_schema is not None:
        return domain_schema
    if not convenience_used:
        return None
    return DomainSchema(
        entity_types=list(entities or []),
        relation_types=list(relations or []),
        instructions=domain_instructions,
        structured_models=list(extraction_models or []),
    )
