import agentic_chunker as ac
import agentic_chunker._domain as domain_mod
from agentic_chunker import (
    AgenticChunker,
    DomainExtractionResult,
    DomainExtractor,
    DomainSchema,
    Entity,
    LlmConfig,
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

    monkeypatch.setattr(ac, "_group_units", fake_group_units)
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
            )

    monkeypatch.setattr(ac, "_group_units", fake_group_units)

    chunker = AgenticChunker(llm=CFG, domain_extractor=ArchitectureExtractor())
    chunks = chunker.chunk("Service A calls Service B.")

    assert chunker.domain_extraction.entities[0].name == "Service A"
    assert chunker.domain_extraction.triples[0].predicate == "CALLS"
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
