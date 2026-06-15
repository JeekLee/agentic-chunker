"""Batch grouping of propositions into chunks.

Propositions are partitioned by header (section); each section is split into
contiguous windows of at most ``window_size``. One LLM ``group`` call per window
clusters its propositions into chunks (returning index clusters with refreshed
title/summary/keywords). Windows run in parallel. A ``max_props`` post-cap splits
oversized clusters; unassigned propositions and grouping failures fall back to
one chunk per proposition so content is never dropped.
"""
from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from ._common import Chunk, Proposition
from .llm import LlmConfig
from .llm import chat_json as _real_chat_json

_PROMPT = """\
당신은 RAG용 청크를 구성하는 편집자입니다.
아래 번호가 매겨진 명제 목록을 의미적으로 일관된 청크로 묶어 주세요.
같은 주제의 명제끼리 한 청크로 모으고, 주제가 다르면 다른 청크로 나눕니다.

명제 목록:
{props}

다음 JSON 배열만 출력하세요(설명 없이):
[{"proposition_indices": [정수, ...], "title": "청크 제목",
  "summary": "청크 한 줄 요약", "keywords": ["키워드", ...]}, ...]
- proposition_indices는 위 목록의 번호(0부터 시작)입니다.
- 모든 명제를 하나 이상의 청크에 포함시키세요.
- title/summary/keywords는 해당 청크 내용을 반영합니다."""


def _default_group(prop_texts: list[str], cfg: LlmConfig | None):
    numbered = "\n".join(f"{i}: {t}" for i, t in enumerate(prop_texts))
    payload = _real_chat_json(_PROMPT.replace("{props}", numbered), cfg)
    return payload if isinstance(payload, list) else None


def _windows(props: list[Proposition], window_size: int) -> list[list[Proposition]]:
    """Partition by header (first-seen section order), then into contiguous windows."""
    order: list[str | None] = []
    by_section: dict[str | None, list[Proposition]] = {}
    for p in props:
        if p.header not in by_section:
            by_section[p.header] = []
            order.append(p.header)
        by_section[p.header].append(p)

    size = max(1, window_size)
    result: list[list[Proposition]] = []
    for header in order:
        section = by_section[header]
        for i in range(0, len(section), size):
            result.append(section[i:i + size])
    return result


def _dedupe_spans(props: list[Proposition]) -> list[tuple[int, int]]:
    seen: list[tuple[int, int]] = []
    for p in props:
        span = (p.char_start, p.char_end)
        if span not in seen:
            seen.append(span)
    return seen


def _own_chunks(props: list[Proposition]) -> list[dict]:
    """Fallback: one chunk per proposition (used for failures / unassigned)."""
    return [
        {"props": [p], "title": p.text[:40], "summary": p.text, "keywords": []}
        for p in props
    ]


def _group_window(
    window: list[Proposition],
    cfg: LlmConfig | None,
    group: Callable[[list[str], LlmConfig | None], list | None],
    max_props: int,
) -> list[dict]:
    """Return a list of chunk-dicts: {props, title, summary, keywords}."""
    texts = [p.text for p in window]
    clusters = group(texts, cfg)
    if not isinstance(clusters, list):
        return _own_chunks(window)

    chunk_dicts: list[dict] = []
    assigned: set[int] = set()
    for cl in clusters:
        if not isinstance(cl, dict):
            continue
        idxs = cl.get("proposition_indices")
        if not isinstance(idxs, list):
            continue
        members: list[Proposition] = []
        for i in idxs:
            if isinstance(i, int) and 0 <= i < len(window) and i not in assigned:
                assigned.add(i)
                members.append(window[i])
        if not members:
            continue
        title = cl.get("title") if isinstance(cl.get("title"), str) else ""
        summary = cl.get("summary") if isinstance(cl.get("summary"), str) else ""
        kw = cl.get("keywords")
        keywords = [k for k in kw if isinstance(k, str)] if isinstance(kw, list) else []
        capped = max(1, max_props)
        for j in range(0, len(members), capped):
            chunk_dicts.append({
                "props": members[j:j + capped],
                "title": title, "summary": summary, "keywords": keywords,
            })

    leftover = [window[i] for i in range(len(window)) if i not in assigned]
    chunk_dicts.extend(_own_chunks(leftover))

    if not chunk_dicts:
        return _own_chunks(window)
    return chunk_dicts


def assign(
    props: list[Proposition],
    cfg: LlmConfig | None,
    *,
    group=_default_group,
    max_props: int = 10,
    window_size: int = 40,
    concurrency: int = 8,
) -> list[Chunk]:
    """Group propositions into Chunks via batched LLM calls (one per window)."""
    if not props:
        return []

    windows = _windows(props, window_size)

    def run(window):
        return _group_window(window, cfg, group, max_props)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        per_window = list(ex.map(run, windows))

    chunks: list[Chunk] = []
    idx = 0
    for window_chunks in per_window:
        for cd in window_chunks:
            members = cd["props"]
            chunks.append(Chunk(
                index=idx,
                text="\n".join(p.text for p in members),
                title=cd["title"],
                summary=cd["summary"],
                keywords=cd["keywords"],
                source_spans=_dedupe_spans(members),
            ))
            idx += 1
    return chunks
