"""Top-level chunking result objects."""
from __future__ import annotations

from dataclasses import dataclass, field

from ._models import Chunk, DocumentGraph
from ._domain_models import DomainExtractionResult, Entity, Triple
from ._structured import StructuredExtraction


@dataclass
class ChunkingResult:
    """Full result for document chunking plus optional domain extraction.

    ``AgenticChunker.chunk()`` intentionally keeps returning ``list[Chunk]`` for
    compatibility. New callers that need the full document graph and extraction
    payload should use ``AgenticChunker.chunk_document()``.
    """

    chunks: list[Chunk] = field(default_factory=list)
    document_graph: DocumentGraph = field(default_factory=DocumentGraph)
    entities: list[Entity] = field(default_factory=list)
    triples: list[Triple] = field(default_factory=list)
    structured_extractions: list[StructuredExtraction] = field(default_factory=list)

    @classmethod
    def from_domain_result(
        cls,
        *,
        chunks: list[Chunk],
        document_graph: DocumentGraph,
        domain_extraction: DomainExtractionResult,
    ) -> "ChunkingResult":
        return cls(
            chunks=chunks,
            document_graph=document_graph,
            entities=list(domain_extraction.entities),
            triples=list(domain_extraction.triples),
            structured_extractions=list(domain_extraction.structured_extractions),
        )

    @property
    def domain_extraction(self) -> DomainExtractionResult:
        return DomainExtractionResult(
            entities=list(self.entities),
            triples=list(self.triples),
            structured_extractions=list(self.structured_extractions),
        )
