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
        max_propositions_per_chunk: post-cap; clusters larger than this are split.
        window_size: propositions per grouping call; larger sections are windowed.
        min_extract_chars: blocks shorter than this skip LLM extraction (emitted verbatim).
        max_concurrency: thread cap for parallel extraction and window grouping.
    """

    def __init__(
        self,
        *,
        llm: LlmConfig,
        max_propositions_per_chunk: int = 10,
        window_size: int = 40,
        min_extract_chars: int = 20,
        max_concurrency: int = 8,
    ) -> None:
        self._llm = llm
        self._max_props = max_propositions_per_chunk
        self._window_size = window_size
        self._min_extract_chars = min_extract_chars
        self._concurrency = max_concurrency

    def chunk(self, markdown: str) -> list[Chunk]:
        blocks = _split(markdown)
        if not blocks:
            return []
        props = _extract(
            blocks,
            self._llm,
            concurrency=self._concurrency,
            min_extract_chars=self._min_extract_chars,
        )
        return _assign(
            props,
            self._llm,
            max_props=self._max_props,
            window_size=self._window_size,
            concurrency=self._concurrency,
        )


__all__ = ["AgenticChunker", "LlmConfig", "Chunk", "Block", "Proposition"]
