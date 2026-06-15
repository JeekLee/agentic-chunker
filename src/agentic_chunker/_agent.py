"""Per-proposition placement agent loop.

Groups propositions within each header (section) independently and in parallel.
For each proposition, `decide()` asks the LLM whether it joins an open chunk or
starts a new one, and returns refreshed title/summary/keywords in the same call.
Chunks at capacity are closed. On any decide failure the proposition starts a new
chunk so content is never dropped.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from ._common import Chunk, Proposition
from .llm import LlmConfig
from .llm import chat_json as _real_chat_json

_PROMPT = """\
당신은 RAG용 청크를 구성하는 편집자입니다.
아래 '새 명제'를 기존 열린 청크 중 하나에 넣을지, 새 청크를 만들지 판단하세요.
같은 주제로 의미가 연결되면 기존 청크에 넣고, 아니면 새로 만듭니다.

열린 청크들(JSON):
{open_chunks}

새 명제:
{prop}

다음 JSON만 출력하세요(설명 없이):
{{"action": "existing" 또는 "new", "chunk_id": 정수 또는 null,
  "title": "청크 제목", "summary": "청크 한 줄 요약", "keywords": ["키워드", ...]}}
action이 existing이면 chunk_id는 위 목록의 id 중 하나여야 합니다.
title/summary/keywords는 명제 반영 후의 갱신된 값입니다."""


def _default_decide(prop_text: str, open_chunks: list[dict], cfg: LlmConfig | None):
    prompt = _PROMPT.format(
        open_chunks=json.dumps(open_chunks, ensure_ascii=False),
        prop=prop_text,
    )
    payload = _real_chat_json(prompt, cfg)
    return payload if isinstance(payload, dict) else None


class _OpenChunk:
    __slots__ = ("id", "props", "title", "summary", "keywords")

    def __init__(self, cid: int):
        self.id = cid
        self.props: list[Proposition] = []
        self.title = ""
        self.summary = ""
        self.keywords: list[str] = []


def _assign_section(props: list[Proposition], cfg, decide, max_props: int) -> list[_OpenChunk]:
    open_chunks: list[_OpenChunk] = []
    next_id = 0
    for p in props:
        candidates = [c for c in open_chunks if len(c.props) < max_props]
        view = [{"id": c.id, "title": c.title, "summary": c.summary} for c in candidates]
        decision = decide(p.text, view, cfg) or {}

        target = None
        if decision.get("action") == "existing":
            cid = decision.get("chunk_id")
            target = next((c for c in candidates if c.id == cid), None)

        if target is None:
            target = _OpenChunk(next_id)
            next_id += 1
            open_chunks.append(target)

        target.props.append(p)
        if decision.get("title"):
            target.title = decision["title"]
        if decision.get("summary"):
            target.summary = decision["summary"]
        kw = decision.get("keywords")
        if isinstance(kw, list):
            target.keywords = [k for k in kw if isinstance(k, str)]
    return open_chunks


def _dedupe_spans(props: list[Proposition]) -> list[tuple[int, int]]:
    seen: list[tuple[int, int]] = []
    for p in props:
        span = (p.char_start, p.char_end)
        if span not in seen:
            seen.append(span)
    return seen


def assign(props, cfg, *, decide=_default_decide, max_props: int = 10, concurrency: int = 8):
    """Group propositions into Chunks. Sections (by header) are independent."""
    if not props:
        return []

    # Preserve first-seen section order.
    order: list[str | None] = []
    by_section: dict[str | None, list[Proposition]] = {}
    for p in props:
        if p.header not in by_section:
            by_section[p.header] = []
            order.append(p.header)
        by_section[p.header].append(p)

    def run(header):
        return _assign_section(by_section[header], cfg, decide, max_props)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        section_results = list(ex.map(run, order))

    chunks: list[Chunk] = []
    idx = 0
    for open_chunks in section_results:
        for oc in open_chunks:
            chunks.append(Chunk(
                index=idx,
                text="\n".join(p.text for p in oc.props),
                title=oc.title,
                summary=oc.summary,
                keywords=oc.keywords,
                source_spans=_dedupe_spans(oc.props),
            ))
            idx += 1
    return chunks
