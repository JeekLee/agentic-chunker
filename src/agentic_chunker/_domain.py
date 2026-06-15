"""Optional domain knowledge extraction over chunks."""
from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json

from ._common import Chunk, DocumentGraph
from .llm import LlmConfig, chat_json


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
class DomainExtractionResult:
    entities: list[Entity] = field(default_factory=list)
    triples: list[Triple] = field(default_factory=list)


@dataclass
class DomainSchema:
    entity_types: list[str]
    relation_types: list[str]
    instructions: str = ""


@dataclass
class DocumentContext:
    chunks: list[Chunk]
    document_graph: DocumentGraph


class DomainExtractor(ABC):
    @abstractmethod
    def extract(self, chunk: Chunk, document_context: DocumentContext) -> DomainExtractionResult:
        pass


def run_domain_extraction(
    chunks: list[Chunk],
    cfg: LlmConfig | None,
    *,
    schema: DomainSchema | None = None,
    extractor: DomainExtractor | None = None,
    concurrency: int = 4,
) -> DomainExtractionResult:
    if not chunks or (schema is None and extractor is None):
        return DomainExtractionResult()
    if schema is not None and extractor is not None:
        raise ValueError("domain_schema and domain_extractor are mutually exclusive")

    selected = extractor or _SchemaDomainExtractor(schema, cfg)
    context = DocumentContext(chunks=chunks, document_graph=_merge_document_graphs(chunks))

    def extract_one(chunk: Chunk) -> DomainExtractionResult:
        try:
            result = selected.extract(chunk, context)
        except Exception:
            result = DomainExtractionResult()
        result = _validate_result(result, schema, chunk)
        chunk.domain_extraction = result
        return result

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        results = list(ex.map(extract_one, chunks))
    return _merge_results(results)


class _SchemaDomainExtractor(DomainExtractor):
    def __init__(self, schema: DomainSchema | None, cfg: LlmConfig | None) -> None:
        self._schema = schema or DomainSchema(entity_types=[], relation_types=[])
        self._cfg = cfg

    def extract(self, chunk: Chunk, document_context: DocumentContext) -> DomainExtractionResult:
        raw = chat_json(_schema_prompt(chunk, self._schema), self._cfg)
        if not isinstance(raw, dict):
            return DomainExtractionResult()
        return _result_from_json(raw, chunk)


def _schema_prompt(chunk: Chunk, schema: DomainSchema) -> str:
    payload = {
        "chunk_id": chunk.id,
        "source": chunk.source,
        "summary": chunk.summary,
        "keywords": chunk.keywords,
        "questions_answered": chunk.questions_answered,
        "entity_types": schema.entity_types,
        "relation_types": schema.relation_types,
        "instructions": schema.instructions,
    }
    return """\
당신은 도메인 지식 그래프 추출기입니다.
청크 원문에서 명시적으로 드러난 entity와 relation triple만 추출하세요.
추론하지 말고, 모든 triple에는 원문 evidence를 포함하세요.

다음 JSON 객체만 출력하세요:
{
  "entities": [{"name": "...", "type": "...", "canonical_name": null, "metadata": {}}],
  "triples": [{"subject": "...", "predicate": "...", "object": "...",
               "evidence": "...", "source_chunk_id": "...",
               "confidence": null, "metadata": {}}]
}

입력:
""" + json.dumps(payload, ensure_ascii=False, indent=2)


def _result_from_json(raw: dict, chunk: Chunk) -> DomainExtractionResult:
    entities: list[Entity] = []
    for item in raw.get("entities", []):
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        entity_type = item.get("type")
        if not isinstance(name, str) or not isinstance(entity_type, str):
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        canonical = item.get("canonical_name")
        entities.append(Entity(
            name=name,
            type=entity_type,
            canonical_name=canonical if isinstance(canonical, str) else None,
            metadata=metadata,
        ))

    triples: list[Triple] = []
    for item in raw.get("triples", []):
        if not isinstance(item, dict):
            continue
        subject = item.get("subject")
        predicate = item.get("predicate")
        obj = item.get("object")
        evidence = item.get("evidence")
        if not all(isinstance(v, str) and v for v in [subject, predicate, obj, evidence]):
            continue
        confidence = item.get("confidence")
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        triples.append(Triple(
            subject=subject,
            predicate=predicate,
            object=obj,
            evidence=evidence,
            source_chunk_id=chunk.id,
            confidence=confidence if isinstance(confidence, int | float) else None,
            metadata=metadata,
        ))
    return DomainExtractionResult(entities=entities, triples=triples)


def _validate_result(result: DomainExtractionResult, schema: DomainSchema | None, chunk: Chunk) -> DomainExtractionResult:
    entity_types = set(schema.entity_types) if schema else set()
    relation_types = set(schema.relation_types) if schema else set()

    entities = [
        entity for entity in result.entities
        if not entity_types or entity.type in entity_types
    ]
    triples: list[Triple] = []
    for triple in result.triples:
        if relation_types and triple.predicate not in relation_types:
            continue
        if not triple.source_chunk_id:
            triple.source_chunk_id = chunk.id
        triples.append(triple)
    return DomainExtractionResult(entities=entities, triples=triples)


def _merge_results(results: list[DomainExtractionResult]) -> DomainExtractionResult:
    entities: list[Entity] = []
    triples: list[Triple] = []
    entity_keys: set[tuple[str, str, str | None]] = set()
    triple_keys: set[tuple[str, str, str, str, str]] = set()

    for result in results:
        for entity in result.entities:
            key = (entity.name, entity.type, entity.canonical_name)
            if key not in entity_keys:
                entity_keys.add(key)
                entities.append(entity)
        for triple in result.triples:
            key = (triple.subject, triple.predicate, triple.object, triple.evidence, triple.source_chunk_id)
            if key not in triple_keys:
                triple_keys.add(key)
                triples.append(triple)
    return DomainExtractionResult(entities=entities, triples=triples)


def _merge_document_graphs(chunks: list[Chunk]) -> DocumentGraph:
    nodes = {}
    edges = {}
    for chunk in chunks:
        for node in chunk.document_graph.nodes:
            nodes.setdefault(node.id, node)
        for edge in chunk.document_graph.edges:
            edges.setdefault((edge.source_id, edge.target_id, edge.type), edge)
    return DocumentGraph(nodes=list(nodes.values()), edges=list(edges.values()))
