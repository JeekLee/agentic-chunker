"""agentic-chunker — proposition-based agentic chunking for RAG.

    from agentic_chunker import AgenticChunker, LlmConfig, Chunk

    chunker = AgenticChunker(
        llm=LlmConfig(url="http://localhost:10080/v1", api_key="...", model="qwen3-..."),
    )
    chunks = chunker.chunk(markdown_text)
"""
from __future__ import annotations

from ._agent import assign as _assign
from ._common import Block, Chunk, Proposition
from ._propositions import extract as _extract
from ._split import split as _split
from .llm import LlmConfig


class AgenticChunker:
    """Markdown/text → semantically coherent chunks via proposition-based agentic chunking.

    Args:
        llm: LlmConfig for an OpenAI-compatible endpoint.
        max_propositions_per_chunk: soft cap aligning chunks to the ~100-200 word sweet spot.
        max_concurrency: thread cap for parallel extraction and section assignment.
    """

    def __init__(
        self,
        *,
        llm: LlmConfig,
        max_propositions_per_chunk: int = 10,
        max_concurrency: int = 8,
    ) -> None:
        self._llm = llm
        self._max_props = max_propositions_per_chunk
        self._concurrency = max_concurrency

    def chunk(self, markdown: str) -> list[Chunk]:
        blocks = _split(markdown)
        if not blocks:
            return []
        props = _extract(blocks, self._llm, concurrency=self._concurrency)
        return _assign(
            props,
            self._llm,
            max_props=self._max_props,
            concurrency=self._concurrency,
        )


__all__ = ["AgenticChunker", "LlmConfig", "Chunk", "Block", "Proposition"]
