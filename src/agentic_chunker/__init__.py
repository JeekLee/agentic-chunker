"""agentic-chunker — proposition-based agentic chunking for RAG.

    from agentic_chunker import AgenticChunker, LlmConfig, Chunk

    chunker = AgenticChunker(
        llm=LlmConfig(url="http://localhost:10080/v1", api_key="...", model="qwen3-..."),
    )
    chunks = chunker.chunk(markdown_text)

Implementation pending — see docs/superpowers/specs/2026-06-15-agentic-chunker-design.md
"""
from __future__ import annotations
