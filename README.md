# agentic-chunker

Proposition-based **agentic chunking** for RAG pipelines, in pure Python.

Takes Markdown (e.g. the output of [md-converter](https://github.com/JeekLee/md-converter)) or plain text and splits it into semantically coherent chunks using an LLM, rather than fixed-size windows. Each chunk carries a generated title, summary, and keywords, plus an approximate mapping back to the source text.

Built on the proposition-based approach validated by *Dense X Retrieval* (propositions outperform passage/sentence chunks on open-domain QA), with a Kamradt-style per-proposition agent loop for placement.

Requires Python 3.11+. Core has **zero runtime dependencies** (stdlib `urllib` for OpenAI-compatible LLM calls).

> Status: in development. See [`docs/superpowers/specs/2026-06-15-agentic-chunker-design.md`](docs/superpowers/specs/2026-06-15-agentic-chunker-design.md) for the design.

## Usage (planned)

```python
from agentic_chunker import AgenticChunker, LlmConfig, Chunk

chunker = AgenticChunker(
    llm=LlmConfig(url="http://localhost:10080/v1", api_key="...", model="qwen3-..."),
)
chunks: list[Chunk] = chunker.chunk(markdown_text)
```

## Pipeline

`md-converter` (HWP/HWPX/PDF → Markdown) → **agentic-chunker** (Markdown → chunks) → embeddings / RAG index.
