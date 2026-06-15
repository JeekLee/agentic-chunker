"""Per-block proposition extraction via LLM, run in parallel.

Each block is sent to the LLM, which returns a JSON array of atomic,
self-contained statements. On any failure the whole block text is kept as a
single proposition so the pipeline never loses content.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from ._common import Block, Proposition
from .llm import LlmConfig
from .llm import chat_json as _real_chat_json

# Instruction prefix; the block text is appended verbatim (no str.format, so
# blocks containing literal "{" / "}" can never break prompt construction).
_PROMPT = """\
아래 텍스트를 원자적 명제(proposition) 목록으로 분해해 주세요.
각 명제는 다른 문장에 의존하지 않고 그 자체로 이해되는 하나의 사실이어야 합니다.
대명사는 가리키는 대상으로 풀어서 자기완결적으로 작성하세요.

지침:
- JSON 문자열 배열로만 출력, 설명 없이 (예: ["...", "..."])
- 원문에 없는 내용 추가 금지

텍스트:
"""


def extract(
    blocks: list[Block],
    cfg: LlmConfig | None,
    *,
    chat_json=_real_chat_json,
    concurrency: int = 8,
    min_extract_chars: int = 20,
) -> list[Proposition]:
    """Extract propositions from each block. Returns a flat list in block order.

    Fail-soft: on any error or empty result for a block, the whole block text is
    kept as a single proposition. Blocks shorter than ``min_extract_chars`` skip
    the LLM entirely and are emitted verbatim (avoids hallucinating context for
    bare labels / headings).
    """
    def one(block: Block) -> list[Proposition]:
        if len(block.text) < min_extract_chars:
            texts = [block.text]
        else:
            texts = []
            try:
                raw = chat_json(_PROMPT + block.text, cfg)
                if isinstance(raw, list):
                    for it in raw:
                        if isinstance(it, str) and it.strip():
                            texts.append(it.strip())
            except Exception:
                texts = []
            if not texts:
                texts = [block.text]  # fallback: keep the whole block
        return [
            Proposition(
                text=t,
                char_start=block.char_start,
                char_end=block.char_end,
                header=block.header,
            )
            for t in texts
        ]

    if not blocks:
        return []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        per_block = list(ex.map(one, blocks))
    return [p for sub in per_block for p in sub]
