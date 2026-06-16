# agentic-chunker — Graph Architecture Direction

**Date:** 2026-06-15
**Status:** Direction approved; core API implemented incrementally
**Related:** [`2026-06-15-agentic-chunker-design.md`](2026-06-15-agentic-chunker-design.md), [`2026-06-15-agentic-chunker-batch-v2-design.md`](2026-06-15-agentic-chunker-batch-v2-design.md)

## Positioning

`agentic-chunker` is not a GraphRAG framework. It is the document understanding
and chunking stage that produces source-preserving chunks plus graph metadata that
downstream RAG or GraphRAG systems can use.

The library should remain domain-agnostic. It must work for HIRA notices, legal
documents, contracts, architecture documents, codebase docs, and other structured
documents without hardcoding any of those domains.

Core goals:

1. Parse documents into structural evidence units.
2. Perform agentic chunking.
3. Preserve document structure as a generic `DocumentGraph`.
4. Orchestrate optional user-defined domain extraction.
5. Validate and serialize the resulting chunks and extraction outputs.

## Boundaries

The library owns document structure. The user owns domain meaning.

Core responsibilities:

```text
Document Parsing
        ↓
Evidence Unit Generation
        ↓
Chunk Generation
        ↓
DocumentGraph Generation
        ↓
Domain Extraction Orchestration
        ↓
Validation
        ↓
Result Serialization
```

Non-goals:

```text
Healthcare ontology
Legal ontology
Insurance business logic
Architecture-specific semantics
Domain-specific reasoning rules
```

Names such as `CoverageRule`, `FeeRule`, `LegalObligation`, or
`ServiceDependency` belong to user-defined extraction schemas, not to the core
library.

## Current Source Layout

The implementation keeps the public API thin and separates internal modules by
responsibility:

```text
src/agentic_chunker/
  __init__.py              public API re-exports only
  _chunker.py              top-level orchestration
  _models.py               generic document, graph, and chunk dataclasses
  _split.py                deterministic Markdown/text block splitting
  _tables.py               Markdown table parsing helpers
  _units.py                evidence-unit construction
  _grouping.py             LLM evidence-unit grouping
  _references.py           generic document reference detection
  _graph.py                DocumentGraph construction and local neighborhoods
  _domain_models.py        domain extraction models and user protocols
  _domain.py               schema/custom extractor orchestration
  _structured.py           structured extraction validation and serialization
  _legacy_agent.py         retained legacy grouping implementation
  _legacy_propositions.py  retained legacy proposition extraction
  llm.py                   OpenAI-compatible JSON chat client
```

This layout keeps domain concepts out of the document pipeline while still
making the top-level `AgenticChunker` easy to use.

## DocumentGraph

`DocumentGraph` is owned by the library. It represents document structure and
chunk-adjacent context that is common across domains.

Primary uses:

- Preserve parent document and section context.
- Connect chunks to tables, figures, captions, and code blocks.
- Preserve document order with `NEXT` / `PREVIOUS`.
- Connect references such as `-> table N` to the referenced table node.
- Enable downstream context expansion:

```text
retrieved chunk
    + parent section
    + referenced table
    + nearby chunks
```

### Nodes

Current and target generic node types:

```python
Document
Section
Chunk
Table
Figure
CodeBlock
```

The core may add generic metadata, such as source spans, section paths, display
format, source chunk ownership, table identifiers, or row counts. It should not
add domain-specific facts.

### Edges

Current and target generic edge types:

```python
HAS_SECTION
HAS_CHILD
HAS_CHUNK

NEXT
PREVIOUS

REFERS_TO

HAS_TABLE
HAS_FIGURE
HAS_CAPTION

CONTINUES
```

Example:

```text
Performance results are shown in table 1.

[table 1]
| model | latency |
```

Graph shape:

```text
Chunk A
  REFERS_TO
Table B
```

If `Chunk A` is retrieved, a downstream retriever can include `Table B` and the
parent section without requiring domain knowledge.

## Domain Extraction

Domain extraction is owned by the user. The core provides an orchestration and
validation framework, but it does not define the domain ontology.

The user defines:

- entity types
- relation types
- extraction instructions
- optional structured extraction models
- optional custom extractors

The library should not ship domain profiles such as:

```python
HIRAProfile()
ArchitectureProfile()
LegalProfile()
```

Those profiles imply that the core understands the domain. Instead, users inject
schemas and extractors.

## Current API

The current implementation supports generic entity, triple, and structured
record extraction via `DomainSchema`, plus a fully custom `DomainExtractor`
escape hatch.

```python
from agentic_chunker import AgenticChunker, DomainSchema, LlmConfig

schema = DomainSchema(
    entity_types=["SERVICE", "DATABASE", "KAFKA_TOPIC"],
    relation_types=["CALLS", "PRODUCES", "CONSUMES"],
    instructions="Extract only explicitly stated facts. Every relation needs evidence.",
)

chunker = AgenticChunker(
    llm=LlmConfig(url="http://localhost:10080/v1", api_key="...", model="qwen3-..."),
    max_units_per_chunk=10,
    domain_schema=schema,
)

chunks = chunker.chunk(markdown_text)
domain_result = chunker.domain_extraction
```

Current result shape:

```python
DomainExtractionResult(
    entities=[...],
    triples=[...],
    structured_extractions=[...],
)
```

Every extracted triple should carry evidence and a source chunk id so downstream
systems can trace the claim back to the original source.

## Convenience API

For simpler user ergonomics, convenience fields layer on top of `DomainSchema`
while keeping the same domain-agnostic boundary.

```python
chunker = AgenticChunker(
    llm=llm,
    entities=["MEDICAL_ACT", "DRUG", "DISEASE"],
    relations=["COVERS", "EXCLUDES", "REQUIRES"],
    extraction_models=[CoverageRule],
)
```

This should be treated as syntax over a generic extraction framework, not as a
domain profile.

## Structured Extraction Models

Triples are useful for graph extraction, but some domains need richer records.
Examples include notices, contracts, laws, and policy documents.

Example user-defined model:

```python
class CoverageRule:
    subject: str
    action: str
    conditions: list[str]
    effect: str
    effective_from: str | None
    evidence: str
```

Example extraction output:

```json
{
  "subject": "Specific exam A",
  "action": "COVERED",
  "conditions": ["Patient has condition X", "At most once per month"],
  "effect": "Covered by reimbursement policy",
  "effective_from": "2026-07-01",
  "evidence": "..."
}
```

Structured extraction models are not a reason to hardcode domain concepts into
the library. Runtime validation uses stdlib dataclasses and protocol-style
support for Pydantic-like classes when present, without adding a required
runtime dependency.

## Result Direction

`chunker.chunk()` returns `list[Chunk]` for compatibility, and optional domain
extraction remains available on `chunker.domain_extraction`.

A richer API returns an explicit result object:

```python
result = chunker.chunk_document(markdown_text)

ChunkingResult(
    chunks=[...],
    document_graph=DocumentGraph(...),
    entities=[...],
    triples=[...],
    structured_extractions=[...],
)
```

This preserves `chunk()` for callers that only need chunks while giving graph
and extraction users a single top-level payload.

## Processing Order

The practical processing order is:

```text
Document
    ↓
Parse
    ↓
Evidence Units
    ↓
Agentic Chunking
    ↓
DocumentGraph Generation
    ↓
Prompt Assembly
    ↓
LLM Domain Extraction
    ↓
Schema Validation
    ↓
Serialization
```

`DocumentGraph` generation depends on final chunk ids and chunk boundaries, so it
normally runs after chunking. If source-level graph generation is later added, it
should remain generic and be reconciled with chunk-level graph generation.

## Design Principles

`DocumentGraph`:

- Owned by the library.
- Represents generic document structure.
- Preserves source context and adjacency.
- Connects chunks to referenced tables, figures, captions, and code blocks.
- Supports downstream context expansion.

`Domain Extraction`:

- Owned by the user.
- Defines entity types, relation types, instructions, and structured models.
- Produces evidence-linked domain facts.
- May be implemented through `DomainSchema` or a custom `DomainExtractor`.

`AgenticChunker`:

- Owns chunking, graph generation, extraction orchestration, validation, and
  serialization.
- Does not own domain ontology, healthcare knowledge, legal knowledge, or
  business rules.

This separation keeps the core generic while enabling downstream GraphRAG,
knowledge extraction, and domain-specific reasoning.
