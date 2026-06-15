# agentic-chunker — Design

**Date:** 2026-06-15
**Status:** Approved (design); implementation pending

## Purpose

A Python module that turns a Markdown/plain-text document into semantically coherent
chunks for RAG, using **proposition-based agentic chunking** rather than fixed-size
windows. Each chunk carries an LLM-generated title, summary, and keywords, plus an
approximate mapping back to the source text.

It is the chunking stage of a pipeline:

> `md-converter` (HWP/HWPX/PDF → Markdown) → **agentic-chunker** (Markdown → chunks) → embeddings / RAG index.

The module mirrors `md-converter`'s conventions: `src/` layout, hatchling build,
stdlib-first (no runtime deps), and an OpenAI-compatible LLM client built on `urllib`.

## Background / research basis

- **Proposition-based retrieval works.** *Dense X Retrieval* shows proposition-unit
  chunks beat passage/sentence units across five open-domain QA datasets (Recall@20
  +10.1% for unsupervised retrievers; multi-point EM gains). Sweet spot ≈ 100–200
  words ≈ ~10 propositions per chunk.
- **Agentic chunking is the slowest/most expensive method** and is an **offline
  indexing** step, not a hot-path operation. With paid APIs the per-proposition loop
  is costly; **we target a self-hosted OpenAI-compatible endpoint** (as md-converter
  does), so token cost is effectively zero and **latency is the only real constraint**.
  This justifies using the full-fidelity per-proposition agent loop, with latency
  mitigated by header pre-split + parallel proposition extraction.

## Decisions

| Topic | Decision |
|-------|----------|
| Input | Markdown / plain-text **string** |
| Strategy | **Proposition-based agentic chunking** (Kamradt-style per-proposition placement loop) |
| Optimizations | Markdown-header pre-split + parallel proposition extraction; section-bounded placement |
| Output fields | `text`, `title`, `summary`, `keywords`, `source_spans` (+ `index`) |
| LLM client | Reuse md-converter pattern: `LlmConfig` + `urllib` call to OpenAI-compatible `/chat/completions` |
| Runtime deps | None (stdlib only); `dev` extra = pytest |

## Package structure

```
agentic-chunker/
├── pyproject.toml              # hatchling, name="agentic-chunker", stdlib only
├── README.md
├── .gitignore
├── src/agentic_chunker/
│   ├── __init__.py             # AgenticChunker class + Chunk dataclass + re-exports
│   ├── llm.py                  # LlmConfig + chat() (urllib, OpenAI-compatible)
│   ├── _common.py              # Chunk, Proposition, Block dataclasses
│   ├── _split.py               # Markdown header pre-split → blocks (pure, no LLM)
│   ├── _propositions.py        # block-level proposition extraction (parallel, capped)
│   └── _agent.py               # per-proposition placement loop (title/summary/keywords)
├── tests/
│   ├── test_split.py           # pure logic, no LLM
│   ├── test_propositions.py    # stub LLM
│   ├── test_agent.py           # stub LLM
│   └── test_chunker.py         # end-to-end, stub LLM
└── docs/superpowers/specs/2026-06-15-agentic-chunker-design.md
```

## Public API

```python
from agentic_chunker import AgenticChunker, LlmConfig, Chunk

chunker = AgenticChunker(
    llm=LlmConfig(url="http://localhost:10080/v1", api_key="...", model="qwen3-..."),
    max_propositions_per_chunk=10,   # aligns with the 100–200 word sweet spot
    max_concurrency=8,
)
chunks: list[Chunk] = chunker.chunk(markdown_text)
```

## Data model

```python
@dataclass
class Block:
    """A unit of source text from the structural pre-split."""
    text: str
    char_start: int          # offset into the original source string
    char_end: int
    header: str | None       # nearest preceding Markdown header (section)

@dataclass
class Proposition:
    """An atomic, self-contained statement extracted from a Block."""
    text: str
    char_start: int          # inherited from source Block (approximate)
    char_end: int
    header: str | None

@dataclass
class Chunk:
    index: int
    text: str                            # contributing propositions, joined
    title: str
    summary: str
    keywords: list[str]
    source_spans: list[tuple[int, int]]  # char offsets of contributing blocks (approx)
```

`source_spans` is approximate by design: propositions are rewritten/decontextualized,
so exact substring mapping is impossible. We track the source `Block` each proposition
came from and aggregate those `(char_start, char_end)` ranges into the chunk.

## Data flow

1. **`_split.split(markdown) -> list[Block]`** — pure, deterministic, no LLM.
   Parse Markdown into sections by ATX headers (`#`…`######`); within each section
   split into paragraph blocks (blank-line separated). Record char offsets and the
   section header on each block.
2. **`_propositions.extract(blocks, llm, concurrency) -> list[Proposition]`** —
   one LLM call per block, run in parallel under a concurrency cap. Prompt asks for a
   JSON array of atomic factual statements. Each resulting `Proposition` inherits its
   block's char span and header.
3. **`_agent.assign(props, llm, max_props, concurrency) -> list[Chunk]`** — the
   per-proposition placement loop. Maintains a list of open chunks (id, title,
   summary, member propositions). For each proposition, one LLM call returns
   `{action: "existing"|"new", chunk_id, title, summary, keywords}` — placement and
   metadata refresh combined into a single call. **Placement never crosses section
   (header) boundaries**, so sections are independent and their loops run in parallel.
   A chunk exceeding `max_propositions_per_chunk` forces a new chunk.
4. **Finalize** — assign sequential `index`, aggregate `source_spans` per chunk,
   assemble `Chunk` objects in document order.

## LLM call budget

- Extraction: 1 call per block (parallel).
- Placement + metadata refresh: 1 call per proposition (combined; sequential within a
  section, parallel across sections).
- No separate keyword pass.

## Error handling (mirrors md-converter)

LLM failures never crash the pipeline; they log to `stderr` and fall back:
- Extraction failure for a block → treat the whole block text as a single proposition.
- Placement failure for a proposition → start a new chunk containing just that proposition.

## Testing (TDD)

- `_split` is pure logic → tested directly with no LLM (headers, offsets, blank-line splits, edge cases: no headers, nested headers, empty input).
- `_propositions`, `_agent`, and the end-to-end `chunk()` use a **stub LLM** — an
  injectable/monkeypatched `chat()` returning canned responses — for deterministic
  tests. No live endpoint required.
- Cases: proposition JSON parsing + malformed-JSON fallback; placement into existing
  vs new chunk; `max_propositions_per_chunk` enforcement; section-boundary isolation;
  source-span aggregation; LLM-failure fallbacks.

## Out of scope (YAGNI)

- Accepting raw documents (.hwp/.pdf) — input is text only; that's md-converter's job.
- Embedding/vector-store integration — chunks are the output; indexing is downstream.
- Exact (verbatim) source mapping — only approximate block-level spans.
- Non-Markdown structural parsing (HTML, etc.).
