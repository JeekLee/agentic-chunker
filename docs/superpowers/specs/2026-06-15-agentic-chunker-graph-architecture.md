# agentic-chunker — Graph Architecture Direction

**Date:** 2026-06-15
**Status:** Direction approved; incremental implementation
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

The current implementation supports generic entity and triple extraction via
`DomainSchema`, plus a fully custom `DomainExtractor` escape hatch.

```python
from agentic_chunker import AgenticChunker, DomainSchema, LlmConfig

schema = DomainSchema(
    entity_types=["SERVICE", "DATABASE", "KAFKA_TOPIC"],
    relation_types=["CALLS", "PRODUCES", "CONSUMES"],
    instructions="Extract only explicitly stated facts. Every relation needs evidence.",
)

chunker = AgenticChunker(
    llm=LlmConfig(url="http://localhost:10080/v1", api_key="...", model="qwen3-..."),
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
)
```

Every extracted triple should carry evidence and a source chunk id so downstream
systems can trace the claim back to the original source.

## Target API Direction

For simpler user ergonomics, a future API can layer convenience fields on top of
`DomainSchema` while keeping the same domain-agnostic boundary.

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

Structured extraction models are a target direction, not a reason to hardcode
domain concepts into the library. If runtime validation uses Pydantic or another
third-party validator, it should be introduced as an optional dependency or via a
protocol so the stdlib-first core remains intact.

## Result Direction

Today, `chunker.chunk()` returns `list[Chunk]`, and optional domain extraction is
available on `chunker.domain_extraction`.

A future richer API can return an explicit result object:

```python
ChunkingResult(
    chunks=[...],
    document_graph=DocumentGraph(...),
    entities=[...],
    triples=[...],
    structured_extractions=[...],
)
```

This should be added compatibly, for example as a new method, while preserving
`chunk()` for callers that only need chunks.

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
