"""Domain extraction data models and user extension interfaces."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ._models import Chunk, DocumentGraph
from ._structured import StructuredExtraction


@dataclass
class Entity:
    name: str
    type: str
    canonical_name: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Triple:
    subject: str
    predicate: str
    object: str
    evidence: str
    source_chunk_id: str
    confidence: float | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class DomainSchema:
    entity_types: list[str] = field(default_factory=list)
    relation_types: list[str] = field(default_factory=list)
    instructions: str = ""
    structured_models: list[type] = field(default_factory=list)


@dataclass
class DomainExtractionResult:
    entities: list[Entity] = field(default_factory=list)
    triples: list[Triple] = field(default_factory=list)
    structured_extractions: list[StructuredExtraction] = field(default_factory=list)


@dataclass
class DocumentContext:
    chunks: list[Chunk]
    document_graph: DocumentGraph


class DomainExtractor(ABC):
    @abstractmethod
    def extract(self, chunk: Chunk, document_context: DocumentContext) -> DomainExtractionResult:
        pass
