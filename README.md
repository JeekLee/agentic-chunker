# agentic-chunker

Proposition-based **agentic chunking** for RAG pipelines, in pure Python.

Takes Markdown (e.g. the output of [md-converter](https://github.com/JeekLee/md-converter)) or plain text and splits it into semantically coherent chunks using an LLM, rather than fixed-size windows. Each chunk carries a generated title, summary, and keywords, plus an approximate mapping back to the source text.

Built on the proposition-based approach validated by *Dense X Retrieval* (propositions outperform passage/sentence chunks on open-domain QA), with a Kamradt-style per-proposition agent loop for placement.

Requires Python 3.11+. Core has **zero runtime dependencies** (stdlib `urllib` for OpenAI-compatible LLM calls).

See [`docs/superpowers/specs/2026-06-15-agentic-chunker-design.md`](docs/superpowers/specs/2026-06-15-agentic-chunker-design.md) for the design.

## Install

```bash
pip install -e .            # core (stdlib only)
pip install -e ".[dev]"     # + pytest
```

## Usage

```python
from agentic_chunker import AgenticChunker, LlmConfig, Chunk

chunker = AgenticChunker(
    llm=LlmConfig(url="http://localhost:10080/v1", api_key="...", model="qwen3-..."),
    max_propositions_per_chunk=10,   # soft cap (~100-200 word sweet spot)
    max_concurrency=8,               # parallel extraction / section assignment
)
chunks: list[Chunk] = chunker.chunk(markdown_text)

for c in chunks:
    print(c.index, c.title, c.keywords)
    print(c.summary)
    print(c.text)
    print(c.source_spans)            # [(char_start, char_end), ...] into the source
```

Each `Chunk` has: `index`, `text`, `title`, `summary`, `keywords`, `source_spans`.

## How it works

1. **Header pre-split** — the Markdown is split into blocks by ATX headers (sections) and blank lines. Deterministic, no LLM.
2. **Proposition extraction** — each block is sent to the LLM in parallel and decomposed into atomic, self-contained propositions.
3. **Agentic placement** — each proposition is routed by the LLM into an existing chunk or a new one (one call also refreshes the chunk's title/summary/keywords). Placement never crosses a section boundary, so sections are processed in parallel.

Every LLM-touching step is fail-soft: on any error the source content is preserved rather than dropped. Agentic chunking is an **offline indexing** step (seconds per document), not a hot-path operation.

## Pipeline

`md-converter` (HWP/HWPX/PDF → Markdown) → **agentic-chunker** (Markdown → chunks) → embeddings / RAG index.
