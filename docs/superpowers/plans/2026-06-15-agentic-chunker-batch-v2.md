# agentic-chunker v2 (Batch Chunking) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-proposition placement loop with batch grouping (one LLM call per section, large sections split into parallel windows) and skip LLM extraction for short label blocks — turning a ~29 min run on a 2.75KB document into low single-minutes while curbing hallucination.

**Architecture:** Three changes. (1) `_propositions.extract` gains a `min_extract_chars` short-block bypass (no LLM for tiny blocks). (2) `_agent.assign` is rewritten: partition props by section → split each section into contiguous windows of ≤ `window_size` → one parallel LLM `group` call per window returning index clusters → build chunks with a `max_props` post-cap and per-proposition fallback. (3) `__init__.py` exposes the new params and drops the per-proposition path. `_common.py`, `_split.py`, `llm.py` are unchanged.

**Tech Stack:** Python 3.11+, stdlib only (`json`, `concurrent.futures`), pytest, hatchling.

**Workflow:** Each task on its own feature branch off `main`, merged via PR (`gh`). IMPORTANT: before any `git push`, run `gh auth switch --user JeekLee` (two gh accounts are logged in; the active one can flip to `CryptoLab-JeekLee` and cause a 403). Commit with `-c user.name='Jeek Lee' -c user.email='sjlee@cryptolab.co.kr'` and end each commit body with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Tests run via `.venv/bin/python -m pytest`.

---

## File Structure

- `src/agentic_chunker/_propositions.py` — MODIFY: add `min_extract_chars` param + short-block bypass.
- `src/agentic_chunker/_agent.py` — REWRITE: batch grouping (`group` seam replaces `decide`; helpers `_default_group`, `_window`, `_build_chunks`).
- `src/agentic_chunker/__init__.py` — MODIFY: add `window_size` / `min_extract_chars` constructor params; forward them; drop per-proposition references.
- `tests/test_propositions.py` — MODIFY: add short-block bypass tests.
- `tests/test_agent.py` — REWRITE: batch-grouping tests (replace per-proposition tests).
- `tests/test_chunker.py` — MODIFY: assert new params forwarded.

---

## Task 1: Short-block extraction bypass (`_propositions.py`)

**Branch:** `feat/v2-short-block-bypass`

**Files:**
- Modify: `src/agentic_chunker/_propositions.py`
- Test: `tests/test_propositions.py`

- [ ] **Step 1: Create the feature branch**

```bash
gh auth switch --user JeekLee
git checkout main && git pull --ff-only && git checkout -b feat/v2-short-block-bypass
```

- [ ] **Step 2: Write the failing tests** (append to `tests/test_propositions.py`)

```python
def test_short_block_skips_llm_and_is_verbatim():
    called = {"n": 0}

    def fake_chat_json(prompt, cfg):
        called["n"] += 1
        return ["should not be used"]

    short = Block(text="목적", char_start=0, char_end=2, header="H")
    props = extract([short], cfg=None, chat_json=fake_chat_json, min_extract_chars=20)
    assert called["n"] == 0                      # no LLM call for a short block
    assert [p.text for p in props] == ["목적"]   # emitted verbatim
    assert props[0].char_start == 0 and props[0].char_end == 2 and props[0].header == "H"


def test_block_at_threshold_length_is_extracted():
    # len("x" * 20) == 20, which is NOT < 20, so it must be extracted.
    called = {"n": 0}

    def fake_chat_json(prompt, cfg):
        called["n"] += 1
        return ["p1", "p2"]

    blk = Block(text="x" * 20, char_start=0, char_end=20, header=None)
    props = extract([blk], cfg=None, chat_json=fake_chat_json, min_extract_chars=20)
    assert called["n"] == 1
    assert [p.text for p in props] == ["p1", "p2"]


def test_min_extract_chars_defaults_to_20():
    called = {"n": 0}

    def fake_chat_json(prompt, cfg):
        called["n"] += 1
        return ["x"]

    # 19-char block, default threshold -> skipped
    blk = Block(text="a" * 19, char_start=0, char_end=19, header=None)
    props = extract([blk], cfg=None, chat_json=fake_chat_json)
    assert called["n"] == 0
    assert [p.text for p in props] == ["a" * 19]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_propositions.py -k "short_block or threshold or defaults_to_20" -v`
Expected: FAIL — `extract()` got an unexpected keyword argument `min_extract_chars`.

- [ ] **Step 4: Modify the implementation**

In `src/agentic_chunker/_propositions.py`, change the `extract` signature and add the bypass at the top of the inner `one(block)` function. The full updated function:

```python
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
```

- [ ] **Step 5: Run the full propositions suite**

Run: `.venv/bin/python -m pytest tests/test_propositions.py -v`
Expected: PASS (the original 7 tests + 3 new = 10).

- [ ] **Step 6: Commit, push, open PR**

```bash
git add src/agentic_chunker/_propositions.py tests/test_propositions.py
git commit -m "feat: skip LLM extraction for short blocks (curb label hallucination)"
gh auth switch --user JeekLee
git push -u origin feat/v2-short-block-bypass
gh pr create --fill --base main
```
Do NOT merge — the controller reviews and merges.

---

## Task 2: Batch grouping rewrite (`_agent.py`)

**Branch:** `feat/v2-batch-agent`

**Files:**
- Modify (rewrite): `src/agentic_chunker/_agent.py`
- Test (rewrite): `tests/test_agent.py`

This replaces the per-proposition `decide` loop with batch grouping. `assign` keeps its
name and return type; the injected seam is now `group(prop_texts, cfg) -> list[dict] | None`
where each dict is `{"proposition_indices": [int], "title": str, "summary": str, "keywords": [str]}`
and indices refer to positions in the passed `prop_texts` list.

- [ ] **Step 1: Create the feature branch**

```bash
gh auth switch --user JeekLee
git checkout main && git pull --ff-only && git checkout -b feat/v2-batch-agent
```

- [ ] **Step 2: Replace the test file** — overwrite `tests/test_agent.py` with:

```python
from agentic_chunker._common import Proposition
from agentic_chunker._agent import assign


def P(text, header, start=0, end=10):
    return Proposition(text=text, char_start=start, char_end=end, header=header)


def test_single_call_clusters_one_section():
    props = [P("Cats purr.", "Cats", 0, 10), P("Cats sleep.", "Cats", 0, 10),
             P("Dogs bark.", "Cats", 20, 30)]
    calls = []

    def fake_group(texts, cfg):
        calls.append(list(texts))
        # cluster first two together, third alone
        return [
            {"proposition_indices": [0, 1], "title": "Cats", "summary": "About cats.", "keywords": ["cats"]},
            {"proposition_indices": [2], "title": "Dogs", "summary": "About dogs.", "keywords": ["dogs"]},
        ]

    chunks = assign(props, cfg=None, group=fake_group)
    assert len(calls) == 1                                    # one section -> one call
    assert calls[0] == ["Cats purr.", "Cats sleep.", "Dogs bark."]
    assert [c.text for c in chunks] == ["Cats purr.\nCats sleep.", "Dogs bark."]
    assert [c.index for c in chunks] == [0, 1]
    assert chunks[0].title == "Cats" and chunks[0].keywords == ["cats"]
    assert chunks[0].source_spans == [(0, 10)]                # both members share (0,10), deduped
    assert chunks[1].source_spans == [(20, 30)]


def test_sections_grouped_independently_in_order():
    props = [P("Alpha.", "A", 0, 6), P("Beta.", "B", 10, 15)]

    def fake_group(texts, cfg):
        return [{"proposition_indices": [0], "title": "t", "summary": "s", "keywords": []}]

    chunks = assign(props, cfg=None, group=fake_group)
    assert [c.text for c in chunks] == ["Alpha.", "Beta."]    # section order preserved
    assert [c.index for c in chunks] == [0, 1]


def test_large_section_is_split_into_windows():
    props = [P(f"f{i}", "S", 0, 2) for i in range(5)]
    seen = []

    def fake_group(texts, cfg):
        seen.append(list(texts))
        return [{"proposition_indices": list(range(len(texts))),
                 "title": "t", "summary": "s", "keywords": []}]

    chunks = assign(props, cfg=None, group=fake_group, window_size=2, max_props=100)
    # 5 props, window 2 -> windows of [f0,f1],[f2,f3],[f4]
    assert seen == [["f0", "f1"], ["f2", "f3"], ["f4"]]
    assert [c.text for c in chunks] == ["f0\nf1", "f2\nf3", "f4"]
    assert [c.index for c in chunks] == [0, 1, 2]


def test_max_props_post_cap_splits_oversized_cluster():
    props = [P(f"f{i}", "S", 0, 2) for i in range(5)]

    def fake_group(texts, cfg):
        return [{"proposition_indices": [0, 1, 2, 3, 4], "title": "t", "summary": "s", "keywords": []}]

    chunks = assign(props, cfg=None, group=fake_group, window_size=100, max_props=2)
    # one cluster of 5, cap 2 -> [f0,f1],[f2,f3],[f4]
    assert [c.text for c in chunks] == ["f0\nf1", "f2\nf3", "f4"]
    assert chunks[0].title == "t" and chunks[2].title == "t"   # title carried to each sub-chunk


def test_invalid_and_duplicate_indices_dropped():
    props = [P("a", "S", 0, 2), P("b", "S", 4, 6)]

    def fake_group(texts, cfg):
        return [{"proposition_indices": [0, 0, 99], "title": "t", "summary": "s", "keywords": []}]

    chunks = assign(props, cfg=None, group=fake_group)
    # index 0 once (dup dropped), 99 out-of-range dropped -> cluster = [a]; b unassigned -> own chunk
    assert [c.text for c in chunks] == ["a", "b"]


def test_unassigned_proposition_becomes_own_chunk():
    props = [P("a", "S", 0, 2), P("b", "S", 4, 6)]

    def fake_group(texts, cfg):
        return [{"proposition_indices": [0], "title": "t", "summary": "s", "keywords": []}]

    chunks = assign(props, cfg=None, group=fake_group)
    assert [c.text for c in chunks] == ["a", "b"]
    assert chunks[1].text == "b"                               # fallback chunk for unassigned


def test_group_failure_falls_back_to_one_chunk_per_proposition():
    props = [P("a", "S", 0, 2), P("b", "S", 4, 6)]

    def fake_group(texts, cfg):
        return None                                           # simulate failure

    chunks = assign(props, cfg=None, group=fake_group)
    assert [c.text for c in chunks] == ["a", "b"]
    assert chunks[0].title == "a" and chunks[0].summary == "a"  # title=text[:40], summary=text


def test_empty_props_returns_empty():
    assert assign([], cfg=None, group=lambda t, c: []) == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_agent.py -v`
Expected: FAIL — `assign()` got an unexpected keyword argument `group` (old signature uses `decide`).

- [ ] **Step 4: Rewrite the implementation** — overwrite `src/agentic_chunker/_agent.py` with:

```python
"""Batch grouping of propositions into chunks.

Propositions are partitioned by header (section); each section is split into
contiguous windows of at most ``window_size``. One LLM ``group`` call per window
clusters its propositions into chunks (returning index clusters with refreshed
title/summary/keywords). Windows run in parallel. A ``max_props`` post-cap splits
oversized clusters; unassigned propositions and grouping failures fall back to
one chunk per proposition so content is never dropped.
"""
from __future__ import annotations

import json
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


def _group_window(window: list[Proposition], cfg, group, max_props: int) -> list[dict]:
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
        # max_props post-cap: split oversized clusters into consecutive sub-chunks.
        for j in range(0, len(members), max(1, max_props)):
            chunk_dicts.append({
                "props": members[j:j + max_props],
                "title": title, "summary": summary, "keywords": keywords,
            })

    # Any proposition the model left unassigned becomes its own chunk.
    leftover = [window[i] for i in range(len(window)) if i not in assigned]
    chunk_dicts.extend(_own_chunks(leftover))

    if not chunk_dicts:                      # nothing valid came back
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

> Note: `_PROMPT` uses `.replace("{props}", ...)` (not `str.format`) so proposition text
> containing literal `{` / `}` cannot break prompt construction — same brace-safety choice
> as the extraction prompt.

- [ ] **Step 5: Run the agent suite**

Run: `.venv/bin/python -m pytest tests/test_agent.py -v`
Expected: PASS (8 tests).

- [ ] **Step 6: Commit, push, open PR**

```bash
git add src/agentic_chunker/_agent.py tests/test_agent.py
git commit -m "feat: replace per-proposition loop with batch window grouping"
gh auth switch --user JeekLee
git push -u origin feat/v2-batch-agent
gh pr create --fill --base main
```
Do NOT merge — the controller reviews and merges.

---

## Task 3: Wire new params into the public API (`__init__.py`)

**Branch:** `feat/v2-public-api`

**Files:**
- Modify: `src/agentic_chunker/__init__.py`
- Test: `tests/test_chunker.py`

- [ ] **Step 1: Create the feature branch**

```bash
gh auth switch --user JeekLee
git checkout main && git pull --ff-only && git checkout -b feat/v2-public-api
```

- [ ] **Step 2: Write the failing test** (append to `tests/test_chunker.py`)

```python
def test_new_params_forwarded_to_stages(monkeypatch):
    captured = {}

    def fake_extract(blocks, cfg, **kw):
        from agentic_chunker._common import Proposition
        captured["min_extract_chars"] = kw.get("min_extract_chars")
        captured["extract_concurrency"] = kw.get("concurrency")
        return [Proposition(b.text, b.char_start, b.char_end, b.header) for b in blocks]

    def fake_assign(props, cfg, **kw):
        captured["window_size"] = kw.get("window_size")
        captured["max_props"] = kw.get("max_props")
        captured["assign_concurrency"] = kw.get("concurrency")
        return [Chunk(index=i, text=p.text) for i, p in enumerate(props)]

    monkeypatch.setattr(ac, "_extract", fake_extract)
    monkeypatch.setattr(ac, "_assign", fake_assign)

    chunker = AgenticChunker(
        llm=CFG,
        max_propositions_per_chunk=7,
        window_size=25,
        min_extract_chars=15,
        max_concurrency=3,
    )
    chunker.chunk("# H\n\nAlpha para.")

    assert captured["min_extract_chars"] == 15
    assert captured["extract_concurrency"] == 3
    assert captured["window_size"] == 25
    assert captured["max_props"] == 7
    assert captured["assign_concurrency"] == 3
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_chunker.py::test_new_params_forwarded_to_stages -v`
Expected: FAIL — `AgenticChunker.__init__()` got an unexpected keyword argument `window_size`.

- [ ] **Step 4: Modify the implementation** — replace the `AgenticChunker` class body in `src/agentic_chunker/__init__.py` with:

```python
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
```

(The imports and `__all__` at top/bottom of the file are unchanged.)

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -v`
Expected: ALL pass across every test file (test_common, test_llm, test_split, test_propositions, test_agent, test_chunker).

- [ ] **Step 6: Commit, push, open PR**

```bash
git add src/agentic_chunker/__init__.py tests/test_chunker.py
git commit -m "feat: expose window_size and min_extract_chars on AgenticChunker"
gh auth switch --user JeekLee
git push -u origin feat/v2-public-api
gh pr create --fill --base main
```
Do NOT merge — the controller reviews and merges.

---

## Self-Review notes (for the planner; not execution steps)

- **Spec coverage:** short-block bypass + `min_extract_chars` (Task 1); batch grouping rewrite with section partition, contiguous windows, parallel `group` calls, index validation, unassigned→own-chunk, group-failure fallback, `max_props` post-cap, source-span dedup, section/window ordering (Task 2); new public params `window_size`/`min_extract_chars` forwarded and per-proposition path dropped (Task 3); `Chunk` schema unchanged (Tasks 2-3); brace-safe prompt via `.replace` (Task 2); fail-soft throughout (Tasks 1-2); stdlib-only (all). Verification (re-run benchmark) is a manual post-merge step, not a code task.
- **Placeholder scan:** none — every code step shows complete code; tests are concrete.
- **Type consistency:** `extract(blocks, cfg, *, chat_json, concurrency, min_extract_chars)` and `assign(props, cfg, *, group, max_props, window_size, concurrency) -> list[Chunk]` signatures match what `__init__.chunk()` calls in Task 3 (keyword args `min_extract_chars`/`concurrency` to extract; `max_props`/`window_size`/`concurrency` to assign). `group` returns `list[dict] | None`; chunk-dict shape `{props, title, summary, keywords}` is internal to `_agent` and consistent between `_group_window`/`_own_chunks`/`assign`. `Chunk` fields match `_common.py`.
