from dataclasses import dataclass

import agentic_chunker as ac
import agentic_chunker._domain as domain_mod
from agentic_chunker import (
    AgenticChunker,
    DomainExtractionResult,
    DomainExtractor,
    DomainSchema,
    Entity,
    LlmConfig,
    StructuredExtraction,
    Triple,
)


CFG = LlmConfig(url="http://x/v1", api_key="k", model="m")


def test_domain_schema_runs_llm_extraction_and_filters_to_schema(monkeypatch):
    def fake_group_units(units, cfg, **kw):
        return list(units)

    def fake_chat_json(prompt, cfg):
        return {
            "entities": [
                {"name": "Order Service", "type": "SERVICE"},
                {"name": "Some Team", "type": "TEAM"},
            ],
            "triples": [
                {
                    "subject": "Order Service",
                    "predicate": "PRODUCES",
                    "object": "orders.created",
                    "evidence": "Order Service publishes orders.created.",
                },
                {
                    "subject": "Order Service",
                    "predicate": "OWNS",
                    "object": "Some Team",
                    "evidence": "Some Team owns Order Service.",
                },
            ],
        }

    monkeypatch.setattr("agentic_chunker._chunker._group_units", fake_group_units)
    monkeypatch.setattr(domain_mod, "chat_json", fake_chat_json)
    schema = DomainSchema(
        entity_types=["SERVICE", "KAFKA_TOPIC"],
        relation_types=["PRODUCES"],
        instructions="Extract explicit architecture facts only.",
    )

    chunker = AgenticChunker(llm=CFG, domain_schema=schema)
    chunks = chunker.chunk("Order Service publishes orders.created.")

    assert [e.name for e in chunker.domain_extraction.entities] == ["Order Service"]
    assert [t.predicate for t in chunker.domain_extraction.triples] == ["PRODUCES"]
    assert chunker.domain_extraction.triples[0].source_chunk_id == "chunk:0"
    assert chunks[0].domain_extraction.triples == chunker.domain_extraction.triples


def test_convenience_schema_supports_structured_extraction_models(monkeypatch):
    @dataclass
    class CoverageRule:
        subject: str
        action: str
        conditions: list[str]
        effect: str
        effective_from: str | None
        evidence: str

    captured = {}

    def fake_group_units(units, cfg, **kw):
        return list(units)

    def fake_chat_json(prompt, cfg):
        captured["prompt"] = prompt
        return {
            "entities": [{"name": "검사 A", "type": "MEDICAL_ACT"}],
            "triples": [
                {
                    "subject": "검사 A",
                    "predicate": "COVERS",
                    "object": "상병 X",
                    "evidence": "상병 X 환자에게 검사 A는 급여한다.",
                }
            ],
            "structured_extractions": [
                {
                    "type": "CoverageRule",
                    "data": {
                        "subject": "검사 A",
                        "action": "COVERED",
                        "conditions": ["상병 X 환자"],
                        "effect": "요양급여 인정",
                        "effective_from": None,
                        "evidence": "상병 X 환자에게 검사 A는 급여한다.",
                    },
                    "evidence": "상병 X 환자에게 검사 A는 급여한다.",
                }
            ],
        }

    monkeypatch.setattr("agentic_chunker._chunker._group_units", fake_group_units)
    monkeypatch.setattr(domain_mod, "chat_json", fake_chat_json)

    result = AgenticChunker(
        llm=CFG,
        entities=["MEDICAL_ACT"],
        relations=["COVERS"],
        extraction_models=[CoverageRule],
        domain_instructions="Only extract explicit coverage rules.",
    ).chunk_document("상병 X 환자에게 검사 A는 급여한다.")

    assert "CoverageRule" in captured["prompt"]
    assert [entity.type for entity in result.entities] == ["MEDICAL_ACT"]
    assert [triple.predicate for triple in result.triples] == ["COVERS"]
    assert len(result.structured_extractions) == 1
    extraction = result.structured_extractions[0]
    assert extraction.type == "CoverageRule"
    assert extraction.data["subject"] == "검사 A"
    assert extraction.source_chunk_id == "chunk:0"
    assert result.domain_extraction.structured_extractions == result.structured_extractions


def test_custom_domain_extractor_is_orchestrated(monkeypatch):
    def fake_group_units(units, cfg, **kw):
        return list(units)

    class ArchitectureExtractor(DomainExtractor):
        def extract(self, chunk, document_context):
            return DomainExtractionResult(
                entities=[Entity(name="Service A", type="SERVICE")],
                triples=[
                    Triple(
                        subject="Service A",
                        predicate="CALLS",
                        object="Service B",
                        evidence="Service A calls Service B.",
                        source_chunk_id=chunk.id,
                    )
                ],
                structured_extractions=[
                    StructuredExtraction(
                        type="ServiceDependency",
                        data={"source": "Service A", "target": "Service B"},
                        evidence="Service A calls Service B.",
                        source_chunk_id=chunk.id,
                    )
                ],
            )

    monkeypatch.setattr("agentic_chunker._chunker._group_units", fake_group_units)

    chunker = AgenticChunker(llm=CFG, domain_extractor=ArchitectureExtractor())
    chunks = chunker.chunk("Service A calls Service B.")

    assert chunker.domain_extraction.entities[0].name == "Service A"
    assert chunker.domain_extraction.triples[0].predicate == "CALLS"
    assert chunker.domain_extraction.structured_extractions[0].type == "ServiceDependency"
    assert chunks[0].domain_extraction.triples[0].source_chunk_id == "chunk:0"


def test_domain_schema_and_extractor_are_mutually_exclusive():
    class NoopExtractor(DomainExtractor):
        def extract(self, chunk, document_context):
            return DomainExtractionResult()

    schema = DomainSchema(entity_types=[], relation_types=[])

    try:
        AgenticChunker(llm=CFG, domain_schema=schema, domain_extractor=NoopExtractor())
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_domain_schema_and_convenience_args_are_mutually_exclusive():
    schema = DomainSchema(entity_types=["SERVICE"], relation_types=[])

    try:
        AgenticChunker(llm=CFG, domain_schema=schema, entities=["DATABASE"])
    except ValueError as exc:
        assert "domain_schema cannot be combined" in str(exc)
    else:
        raise AssertionError("expected ValueError")
