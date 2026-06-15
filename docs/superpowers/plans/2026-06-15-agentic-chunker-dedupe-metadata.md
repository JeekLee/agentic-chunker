# agentic-chunker v2.1 (De-duplicate Chunk Metadata) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the batch grouper from emitting consecutive chunks with identical title/summary/keywords — pass the `max_props` target into the `group` prompt to curb over-merging, and suffix forced post-cap splits with a `(k/N)` part marker so sub-chunks are distinguishable.

**Architecture:** One cohesive change to `_agent.py`: the injected `group` seam gains a `max_props` argument; `_default_group` embeds the size target in the prompt; `_group_window` adds a part marker to each sub-chunk title when a cluster is split into N > 1 parts. All other behavior (windowing, fail-soft, index validation, span dedup, hard cap) is unchanged. No other module changes.

**Tech Stack:** Python 3.11+, stdlib only, pytest, hatchling.

**Workflow:** Single feature branch off `main`, merged via PR. IMPORTANT: before any `git push`, run `gh auth switch --user JeekLee` (two gh accounts logged in; active can flip to `CryptoLab-JeekLee` → 403). Commit with `-c user.name='Jeek Lee' -c user.email='sjlee@cryptolab.co.kr'`, body ending with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Tests run via `.venv/bin/python -m pytest`.

---

## File Structure

- `src/agentic_chunker/_agent.py` — MODIFY: `_PROMPT` (+`{max_props}` placeholder & guidance), `_default_group` (+`max_props` param), `_group_window` (call `group(texts, cfg, max_props)`; part-marker on N>1 splits; `group` Callable type updated). `_windows`, `_dedupe_spans`, `_own_chunks`, `assign` are unchanged.
- `tests/test_agent.py` — MODIFY: all fakes adopt the `group(texts, cfg, max_props)` signature; the post-cap test asserts `(k/N)` titles; add single-part-no-marker and max_props-passed-to-group tests.

---

## Task 1: max_props in prompt + part marker on splits

**Branch:** `feat/v2.1-dedupe-metadata`

**Files:**
- Modify: `src/agentic_chunker/_agent.py`
- Test: `tests/test_agent.py`

- [ ] **Step 1: Create the feature branch**

```bash
gh auth switch --user JeekLee
git checkout main && git pull --ff-only && git checkout -b feat/v2.1-dedupe-metadata
```

- [ ] **Step 2: Replace the test file** — OVERWRITE `tests/test_agent.py` with exactly:

```python
from agentic_chunker._common import Proposition
from agentic_chunker._agent import assign


def P(text, header, start=0, end=10):
    return Proposition(text=text, char_start=start, char_end=end, header=header)


def test_single_call_clusters_one_section():
    props = [P("Cats purr.", "Cats", 0, 10), P("Cats sleep.", "Cats", 0, 10),
             P("Dogs bark.", "Cats", 20, 30)]
    calls = []

    def fake_group(texts, cfg, max_props):
        calls.append(list(texts))
        return [
            {"proposition_indices": [0, 1], "title": "Cats", "summary": "About cats.", "keywords": ["cats"]},
            {"proposition_indices": [2], "title": "Dogs", "summary": "About dogs.", "keywords": ["dogs"]},
        ]

    chunks = assign(props, cfg=None, group=fake_group)
    assert len(calls) == 1
    assert calls[0] == ["Cats purr.", "Cats sleep.", "Dogs bark."]
    assert [c.text for c in chunks] == ["Cats purr.\nCats sleep.", "Dogs bark."]
    assert [c.index for c in chunks] == [0, 1]
    assert chunks[0].title == "Cats" and chunks[0].keywords == ["cats"]
    assert chunks[0].source_spans == [(0, 10)]
    assert chunks[1].source_spans == [(20, 30)]


def test_sections_grouped_independently_in_order():
    props = [P("Alpha.", "A", 0, 6), P("Beta.", "B", 10, 15)]
    calls = []

    def fake_group(texts, cfg, max_props):
        calls.append(list(texts))
        return [{"proposition_indices": [0], "title": "t", "summary": "s", "keywords": []}]

    chunks = assign(props, cfg=None, group=fake_group)
    assert calls == [["Alpha."], ["Beta."]]
    assert [c.text for c in chunks] == ["Alpha.", "Beta."]
    assert [c.index for c in chunks] == [0, 1]


def test_large_section_is_split_into_windows():
    props = [P(f"f{i}", "S", 0, 2) for i in range(5)]
    seen = []

    def fake_group(texts, cfg, max_props):
        seen.append(list(texts))
        return [{"proposition_indices": list(range(len(texts))),
                 "title": "t", "summary": "s", "keywords": []}]

    chunks = assign(props, cfg=None, group=fake_group, window_size=2, max_props=100)
    assert seen == [["f0", "f1"], ["f2", "f3"], ["f4"]]
    assert [c.text for c in chunks] == ["f0\nf1", "f2\nf3", "f4"]
    assert [c.index for c in chunks] == [0, 1, 2]
    assert [c.title for c in chunks] == ["t", "t", "t"]   # single-part windows: no marker


def test_max_props_post_cap_splits_with_part_markers():
    props = [P(f"f{i}", "S", 0, 2) for i in range(5)]

    def fake_group(texts, cfg, max_props):
        return [{"proposition_indices": [0, 1, 2, 3, 4], "title": "t", "summary": "s", "keywords": []}]

    chunks = assign(props, cfg=None, group=fake_group, window_size=100, max_props=2)
    assert [c.text for c in chunks] == ["f0\nf1", "f2\nf3", "f4"]
    # one cluster of 5 split into 3 parts -> distinguishable titles
    assert [c.title for c in chunks] == ["t (1/3)", "t (2/3)", "t (3/3)"]
    # summary/keywords unchanged across parts
    assert all(c.summary == "s" for c in chunks)


def test_single_part_cluster_has_no_marker():
    props = [P("a", "S", 0, 2), P("b", "S", 4, 6)]

    def fake_group(texts, cfg, max_props):
        return [{"proposition_indices": [0, 1], "title": "t", "summary": "s", "keywords": []}]

    chunks = assign(props, cfg=None, group=fake_group, max_props=10)
    assert len(chunks) == 1
    assert chunks[0].title == "t"   # N == 1 -> no (k/N) marker


def test_max_props_is_passed_to_group():
    props = [P("a", "S", 0, 2)]
    seen = {}

    def fake_group(texts, cfg, max_props):
        seen["max_props"] = max_props
        return [{"proposition_indices": [0], "title": "t", "summary": "s", "keywords": []}]

    assign(props, cfg=None, group=fake_group, max_props=7)
    assert seen["max_props"] == 7


def test_invalid_and_duplicate_indices_dropped():
    props = [P("a", "S", 0, 2), P("b", "S", 4, 6)]

    def fake_group(texts, cfg, max_props):
        return [{"proposition_indices": [0, 0, 99], "title": "t", "summary": "s", "keywords": []}]

    chunks = assign(props, cfg=None, group=fake_group)
    assert [c.text for c in chunks] == ["a", "b"]


def test_unassigned_proposition_becomes_own_chunk():
    props = [P("a", "S", 0, 2), P("b", "S", 4, 6)]

    def fake_group(texts, cfg, max_props):
        return [{"proposition_indices": [0], "title": "t", "summary": "s", "keywords": []}]

    chunks = assign(props, cfg=None, group=fake_group)
    assert [c.text for c in chunks] == ["a", "b"]
    assert chunks[1].text == "b"


def test_group_failure_falls_back_to_one_chunk_per_proposition():
    props = [P("a", "S", 0, 2), P("b", "S", 4, 6)]

    def fake_group(texts, cfg, max_props):
        return None

    chunks = assign(props, cfg=None, group=fake_group)
    assert [c.text for c in chunks] == ["a", "b"]
    assert chunks[0].title == "a" and chunks[0].summary == "a"


def test_empty_props_returns_empty():
    assert assign([], cfg=None, group=lambda t, c, m: []) == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_agent.py -v`
Expected: FAIL — the `group` fakes now take 3 args but the current code calls `group(texts, cfg)` (TypeError: missing 1 required positional argument 'max_props'), and the new marker tests fail.

- [ ] **Step 4: Modify the implementation** — OVERWRITE `src/agentic_chunker/_agent.py` with exactly:

```python
"""Batch grouping of propositions into chunks.

Propositions are partitioned by header (section); each section is split into
contiguous windows of at most ``window_size``. One LLM ``group`` call per window
clusters its propositions into chunks (returning index clusters with refreshed
title/summary/keywords). Windows run in parallel. A ``max_props`` post-cap splits
oversized clusters into ``(k/N)``-suffixed sub-chunks; unassigned propositions and
grouping failures fall back to one chunk per proposition so content is never dropped.
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
같은 주제의 명제끼리 한 청크로 모으되, 한 청크에는 명제를 약 {max_props}개 이하로 담고,
주제가 길면 더 잘게 나누세요. 서로 다른 주제는 반드시 다른 청크로 분리합니다.

명제 목록:
{props}

다음 JSON 배열만 출력하세요(설명 없이):
[{"proposition_indices": [정수, ...], "title": "청크 제목",
  "summary": "청크 한 줄 요약", "keywords": ["키워드", ...]}, ...]
- proposition_indices는 위 목록의 번호(0부터 시작)입니다.
- 모든 명제를 하나 이상의 청크에 포함시키세요.
- title/summary/keywords는 해당 청크 내용을 반영합니다."""


def _default_group(prop_texts: list[str], cfg: LlmConfig | None, max_props: int):
    numbered = "\n".join(f"{i}: {t}" for i, t in enumerate(prop_texts))
    prompt = _PROMPT.replace("{max_props}", str(max_props)).replace("{props}", numbered)
    payload = _real_chat_json(prompt, cfg)
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
    group: Callable[[list[str], LlmConfig | None, int], list | None],
    max_props: int,
) -> list[dict]:
    """Return a list of chunk-dicts: {props, title, summary, keywords}."""
    texts = [p.text for p in window]
    clusters = group(texts, cfg, max_props)
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
        parts = [members[j:j + capped] for j in range(0, len(members), capped)]
        n = len(parts)
        for k, part in enumerate(parts, start=1):
            part_title = title if n == 1 else f"{title} ({k}/{n})"
            chunk_dicts.append({
                "props": part,
                "title": part_title, "summary": summary, "keywords": keywords,
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
```

> Brace-safety: `_PROMPT` is a plain string with literal `{` / `}` in the JSON example. Assembly uses `.replace("{max_props}", ...).replace("{props}", ...)` (NOT `str.format`), so the JSON-example braces and any braces in proposition text are never interpreted. Do not convert to an f-string or `.format`.

- [ ] **Step 5: Run the agent suite**

Run: `.venv/bin/python -m pytest tests/test_agent.py -v`
Expected: PASS (10 tests).

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: ALL pass (no regression in other files).

- [ ] **Step 7: Commit, push, open PR**

```bash
git add src/agentic_chunker/_agent.py tests/test_agent.py
git commit -m "feat: de-duplicate chunk metadata (max_props in prompt + (k/N) split markers)"
gh auth switch --user JeekLee
git push -u origin feat/v2.1-dedupe-metadata
gh pr create --fill --base main
```
Do NOT merge — the controller reviews and merges.

---

## Self-Review notes (for the planner; not execution steps)

- **Spec coverage:** `group` seam gains `max_props` (Step 4 signature + `_group_window` call); `_default_group` embeds `{max_props}` via `.replace` (Step 4 prompt + `_default_group`); part marker `(k/N)` on N>1 splits, no marker on N==1 (Step 4 `_group_window` parts loop); hard cap kept (`capped`); summary/keywords unchanged across parts; `_own_chunks` untouched; all invariants (fail-soft, windowing, index validation, span dedup, sequential index) unchanged. Tests cover: marker on split, no marker single-part, max_props passed to group, plus all existing behaviors updated to the new seam. Verification (benchmark re-run) is a manual post-merge step.
- **Placeholder scan:** none — full file contents provided; tests concrete.
- **Type consistency:** `group` seam is `Callable[[list[str], LlmConfig | None, int], list | None]` and is called as `group(texts, cfg, max_props)` in `_group_window`; `_default_group(prop_texts, cfg, max_props)` matches; `assign`'s default `group=_default_group` and `run` → `_group_window(window, cfg, group, max_props)` consistent. `Chunk` fields unchanged. Every test fake uses the 3-arg signature including the empty-case lambda `lambda t, c, m: []`.
