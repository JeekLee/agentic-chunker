# agentic-chunker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pure-Python module that turns Markdown/plain text into semantically coherent RAG chunks via proposition-based agentic chunking, each chunk carrying a title, summary, keywords, and approximate source spans.

**Architecture:** Four stages — (1) deterministic Markdown header pre-split into blocks, (2) parallel per-block proposition extraction via LLM, (3) per-proposition placement agent loop (section-bounded, parallel across sections) that assigns propositions to chunks and refreshes title/summary/keywords in one call, (4) finalize into ordered `Chunk` objects. LLM access is an OpenAI-compatible `urllib` client (mirrors md-converter). Core is stdlib-only.

**Tech Stack:** Python 3.11+, stdlib only (`urllib`, `json`, `re`, `dataclasses`, `concurrent.futures`), pytest for tests, hatchling build.

**Workflow:** Each task is done on its own feature branch off `main`, then merged via a PR (`gh`). Commit after each task; open the PR when the task's tests pass.

---

## File Structure

- `src/agentic_chunker/_common.py` — `Block`, `Proposition`, `Chunk` dataclasses (no logic, no deps).
- `src/agentic_chunker/llm.py` — `LlmConfig` dataclass + `chat()` (text) + `chat_json()` (parse JSON from a chat reply). OpenAI-compatible `urllib` calls.
- `src/agentic_chunker/_split.py` — `split(markdown) -> list[Block]`. Pure, deterministic, no LLM.
- `src/agentic_chunker/_propositions.py` — `extract(blocks, llm, concurrency) -> list[Proposition]`. Parallel per-block LLM extraction with fallback.
- `src/agentic_chunker/_agent.py` — `assign(props, llm, max_props, concurrency) -> list[Chunk]`. Per-proposition placement loop, section-bounded, parallel across sections, with fallback.
- `src/agentic_chunker/__init__.py` — `AgenticChunker` class wiring the four stages; re-exports `LlmConfig`, `Chunk`.
- `tests/test_split.py`, `tests/test_llm.py`, `tests/test_propositions.py`, `tests/test_agent.py`, `tests/test_chunker.py`.

A `StubLlm` test helper (defined inline in each test that needs it) implements the same surface as the real `chat`/`chat_json` calls so tests are deterministic and need no live endpoint.

---

## Task 1: Data model (`_common.py`)

**Branch:** `feat/data-model`

**Files:**
- Create: `src/agentic_chunker/_common.py`
- Test: `tests/test_common.py`

- [ ] **Step 1: Create the feature branch**

```bash
git checkout main && git pull --ff-only && git checkout -b feat/data-model
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_common.py`:

```python
from agentic_chunker._common import Block, Proposition, Chunk


def test_block_holds_text_offsets_and_header():
    b = Block(text="hello", char_start=0, char_end=5, header="Intro")
    assert b.text == "hello"
    assert (b.char_start, b.char_end) == (0, 5)
    assert b.header == "Intro"


def test_proposition_carries_source_span_and_header():
    p = Proposition(text="X is Y.", char_start=10, char_end=20, header="Intro")
    assert p.text == "X is Y."
    assert (p.char_start, p.char_end) == (10, 20)
    assert p.header == "Intro"


def test_chunk_has_all_output_fields_with_defaults():
    c = Chunk(index=0, text="X is Y.")
    assert c.index == 0
    assert c.text == "X is Y."
    assert c.title == ""
    assert c.summary == ""
    assert c.keywords == []
    assert c.source_spans == []
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_common.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentic_chunker._common'`

- [ ] **Step 4: Write minimal implementation**

Create `src/agentic_chunker/_common.py`:

```python
"""Shared dataclasses for agentic-chunker."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Block:
    """A unit of source text from the structural pre-split."""
    text: str
    char_start: int
    char_end: int
    header: str | None = None


@dataclass
class Proposition:
    """An atomic, self-contained statement extracted from a Block."""
    text: str
    char_start: int
    char_end: int
    header: str | None = None


@dataclass
class Chunk:
    """A semantically coherent chunk emitted by the chunker."""
    index: int
    text: str
    title: str = ""
    summary: str = ""
    keywords: list[str] = field(default_factory=list)
    source_spans: list[tuple[int, int]] = field(default_factory=list)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_common.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit, push, open PR**

```bash
git add src/agentic_chunker/_common.py tests/test_common.py
git commit -m "feat: add Block, Proposition, Chunk dataclasses"
git push -u origin feat/data-model
gh pr create --fill --base main
gh pr merge --merge --delete-branch
```

---

## Task 2: LLM client (`llm.py`)

**Branch:** `feat/llm-client`

**Files:**
- Create: `src/agentic_chunker/llm.py`
- Test: `tests/test_llm.py`

- [ ] **Step 1: Create the feature branch**

```bash
git checkout main && git pull --ff-only && git checkout -b feat/llm-client
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_llm.py`. These tests monkeypatch `urllib.request.urlopen` so no network is used.

```python
import io
import json

import agentic_chunker.llm as llm_mod
from agentic_chunker.llm import LlmConfig, chat, chat_json


class _FakeResp:
    def __init__(self, payload: dict):
        self._data = json.dumps(payload).encode()

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch_reply(monkeypatch, content: str, capture: dict | None = None):
    def fake_urlopen(req, timeout=0):
        if capture is not None:
            capture["url"] = req.full_url
            capture["body"] = json.loads(req.data.decode())
            capture["headers"] = req.headers
        return _FakeResp({"choices": [{"message": {"content": content}}]})

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", fake_urlopen)


CFG = LlmConfig(url="http://localhost:10080/v1", api_key="k", model="m")


def test_chat_returns_message_content(monkeypatch):
    _patch_reply(monkeypatch, "hello world")
    assert chat("hi", CFG) == "hello world"


def test_chat_posts_to_chat_completions_with_auth(monkeypatch):
    cap = {}
    _patch_reply(monkeypatch, "ok", cap)
    chat("hi", CFG)
    assert cap["url"] == "http://localhost:10080/v1/chat/completions"
    assert cap["body"]["model"] == "m"
    assert cap["body"]["messages"][0]["content"] == "hi"
    assert cap["headers"]["Authorization"] == "Bearer k"


def test_chat_json_parses_plain_json(monkeypatch):
    _patch_reply(monkeypatch, '[{"a": 1}]')
    assert chat_json("give json", CFG) == [{"a": 1}]


def test_chat_json_strips_code_fence(monkeypatch):
    _patch_reply(monkeypatch, '```json\n{"a": 1}\n```')
    assert chat_json("give json", CFG) == {"a": 1}


def test_chat_json_returns_none_on_garbage(monkeypatch):
    _patch_reply(monkeypatch, "not json at all")
    assert chat_json("give json", CFG) is None
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentic_chunker.llm'`

- [ ] **Step 4: Write minimal implementation**

Create `src/agentic_chunker/llm.py`:

```python
"""OpenAI-compatible LLM client (stdlib urllib).

Mirrors md-converter's llm.py: one POST to /chat/completions, no external deps.
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from dataclasses import dataclass


@dataclass
class LlmConfig:
    url: str        # base URL, e.g. "http://localhost:10080/v1"
    api_key: str
    model: str      # e.g. "qwen3-30b-a3b"
    temperature: float = 0.0
    timeout: int = 60


def chat(prompt: str, cfg: LlmConfig) -> str:
    """Send a single user message, return the assistant text (or "" on failure)."""
    body = json.dumps({
        "model": cfg.model,
        "temperature": cfg.temperature,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    endpoint = f"{cfg.url.rstrip('/')}/chat/completions"
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
            result = json.loads(resp.read())
        return result["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        sys.stderr.write(f"  chat failed: {exc}\n")
        return ""


def chat_json(prompt: str, cfg: LlmConfig):
    """chat() then parse the reply as JSON. Strips ```json fences. None on failure."""
    text = chat(prompt, cfg)
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```\s*$", "", text)
    try:
        return json.loads(text)
    except Exception as exc:
        sys.stderr.write(f"  chat_json parse failed: {exc}\n")
        return None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_llm.py -v`
Expected: PASS (5 passed)

- [ ] **Step 6: Commit, push, open PR**

```bash
git add src/agentic_chunker/llm.py tests/test_llm.py
git commit -m "feat: add OpenAI-compatible LLM client (chat, chat_json)"
git push -u origin feat/llm-client
gh pr create --fill --base main
gh pr merge --merge --delete-branch
```

---

## Task 3: Markdown header pre-split (`_split.py`)

**Branch:** `feat/split`

**Files:**
- Create: `src/agentic_chunker/_split.py`
- Test: `tests/test_split.py`

- [ ] **Step 1: Create the feature branch**

```bash
git checkout main && git pull --ff-only && git checkout -b feat/split
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_split.py`:

```python
from agentic_chunker._split import split


def test_empty_input_yields_no_blocks():
    assert split("") == []
    assert split("   \n\n  ") == []


def test_single_paragraph_no_header():
    blocks = split("Just one paragraph of text.")
    assert len(blocks) == 1
    b = blocks[0]
    assert b.text == "Just one paragraph of text."
    assert b.header is None
    assert b.char_start == 0
    assert b.char_end == len("Just one paragraph of text.")


def test_paragraphs_split_on_blank_lines():
    blocks = split("First para.\n\nSecond para.")
    assert [b.text for b in blocks] == ["First para.", "Second para."]


def test_header_assigned_to_following_blocks():
    md = "# Intro\n\nAlpha para.\n\n## Details\n\nBeta para."
    blocks = split(md)
    texts = [(b.header, b.text) for b in blocks]
    assert texts == [("Intro", "Alpha para."), ("Details", "Beta para.")]


def test_header_line_is_not_emitted_as_a_block():
    blocks = split("# Only A Header\n")
    assert blocks == []


def test_char_offsets_point_into_source():
    md = "# H\n\nHello there."
    blocks = split(md)
    assert len(blocks) == 1
    b = blocks[0]
    assert md[b.char_start:b.char_end] == "Hello there."
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_split.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentic_chunker._split'`

- [ ] **Step 4: Write minimal implementation**

Create `src/agentic_chunker/_split.py`:

```python
"""Deterministic Markdown pre-split into blocks. No LLM.

Splits on ATX headers (# .. ######) to set the section, and on blank lines
into paragraph blocks within each section. Each block records its char
offsets into the original source and its nearest preceding header.
"""
from __future__ import annotations

import re

from ._common import Block

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def split(markdown: str) -> list[Block]:
    blocks: list[Block] = []
    header: str | None = None

    # Walk lines, tracking absolute char offsets. Accumulate non-blank,
    # non-header lines into a paragraph buffer; flush on blank line or header.
    buf: list[str] = []
    buf_start = 0
    pos = 0

    def flush(end: int) -> None:
        nonlocal buf, buf_start
        if not buf:
            return
        text = "\n".join(buf).strip()
        if text:
            start = buf_start
            blocks.append(Block(text=text, char_start=start, char_end=start + len(text), header=header))
        buf = []

    for line in markdown.splitlines(keepends=True):
        stripped = line.strip()
        line_start = pos
        pos += len(line)

        m = _HEADER_RE.match(stripped)
        if m:
            flush(line_start)
            header = m.group(2).strip()
            continue
        if not stripped:
            flush(line_start)
            continue
        if not buf:
            buf_start = line_start
        buf.append(line.rstrip("\n"))

    flush(pos)
    return blocks
```

> Note: `char_start` is the offset of the first non-blank line of the block; `char_end = char_start + len(stripped_text)`. This is an approximate span (trailing whitespace within the block is not re-expanded), which satisfies the spec's "approximate source mapping" requirement and makes `md[start:end]` recover the block text for single-line blocks.

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_split.py -v`
Expected: PASS (6 passed)

- [ ] **Step 6: Commit, push, open PR**

```bash
git add src/agentic_chunker/_split.py tests/test_split.py
git commit -m "feat: add deterministic Markdown header pre-split"
git push -u origin feat/split
gh pr create --fill --base main
gh pr merge --merge --delete-branch
```

---

## Task 4: Proposition extraction (`_propositions.py`)

**Branch:** `feat/propositions`

**Files:**
- Create: `src/agentic_chunker/_propositions.py`
- Test: `tests/test_propositions.py`

- [ ] **Step 1: Create the feature branch**

```bash
git checkout main && git pull --ff-only && git checkout -b feat/propositions
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_propositions.py`. `extract` takes an injectable `chat_json` callable (default = the real one) so tests stay deterministic.

```python
from agentic_chunker._common import Block
from agentic_chunker._propositions import extract


def make_blocks():
    return [
        Block(text="Cats purr. They also sleep.", char_start=0, char_end=27, header="Cats"),
        Block(text="Dogs bark.", char_start=30, char_end=40, header="Dogs"),
    ]


def test_extract_returns_one_proposition_per_returned_item():
    calls = []

    def fake_chat_json(prompt, cfg):
        calls.append(prompt)
        if "Cats purr" in prompt:
            return ["Cats purr.", "Cats sleep."]
        return ["Dogs bark."]

    props = extract(make_blocks(), cfg=None, chat_json=fake_chat_json)
    assert [p.text for p in props] == ["Cats purr.", "Cats sleep.", "Dogs bark."]


def test_extracted_propositions_inherit_block_span_and_header():
    def fake_chat_json(prompt, cfg):
        if "Cats purr" in prompt:
            return ["Cats purr.", "Cats sleep."]
        return ["Dogs bark."]

    props = extract(make_blocks(), cfg=None, chat_json=fake_chat_json)
    cats = [p for p in props if p.header == "Cats"]
    assert all(p.char_start == 0 and p.char_end == 27 for p in cats)
    dogs = [p for p in props if p.header == "Dogs"]
    assert dogs[0].char_start == 30 and dogs[0].char_end == 40


def test_fallback_uses_block_text_when_llm_returns_none():
    def fake_chat_json(prompt, cfg):
        return None

    props = extract(make_blocks(), cfg=None, chat_json=fake_chat_json)
    assert [p.text for p in props] == ["Cats purr. They also sleep.", "Dogs bark."]


def test_non_string_items_are_ignored():
    def fake_chat_json(prompt, cfg):
        return ["good", 123, {"x": 1}, "  "]

    props = extract([make_blocks()[0]], cfg=None, chat_json=fake_chat_json)
    assert [p.text for p in props] == ["good"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_propositions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentic_chunker._propositions'`

- [ ] **Step 4: Write minimal implementation**

Create `src/agentic_chunker/_propositions.py`:

```python
"""Per-block proposition extraction via LLM, run in parallel.

Each block is sent to the LLM, which returns a JSON array of atomic,
self-contained statements. On any failure the whole block text is kept as a
single proposition so the pipeline never loses content.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from ._common import Block, Proposition
from .llm import chat_json as _real_chat_json

_PROMPT = """\
아래 텍스트를 원자적 명제(proposition) 목록으로 분해해 주세요.
각 명제는 다른 문장에 의존하지 않고 그 자체로 이해되는 하나의 사실이어야 합니다.
대명사는 가리키는 대상으로 풀어서 자기완결적으로 작성하세요.

지침:
- JSON 문자열 배열로만 출력, 설명 없이 (예: ["...", "..."])
- 원문에 없는 내용 추가 금지

텍스트:
{content}"""


def extract(blocks, cfg, *, chat_json=_real_chat_json, concurrency: int = 8):
    """Extract propositions from each block. Returns a flat list in block order."""
    def one(block: Block) -> list[Proposition]:
        items = chat_json(_PROMPT.format(content=block.text), cfg)
        if not isinstance(items, list):
            items = None
        texts = []
        if items is not None:
            for it in items:
                if isinstance(it, str) and it.strip():
                    texts.append(it.strip())
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

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_propositions.py -v`
Expected: PASS (4 passed)

- [ ] **Step 6: Commit, push, open PR**

```bash
git add src/agentic_chunker/_propositions.py tests/test_propositions.py
git commit -m "feat: add parallel per-block proposition extraction"
git push -u origin feat/propositions
gh pr create --fill --base main
gh pr merge --merge --delete-branch
```

---

## Task 5: Placement agent loop (`_agent.py`)

**Branch:** `feat/agent`

**Files:**
- Create: `src/agentic_chunker/_agent.py`
- Test: `tests/test_agent.py`

The agent groups propositions **within a single header (section)** at a time. For each
proposition it calls the LLM once, passing the section's currently-open chunks
(id, title, summary) and the proposition. The LLM returns
`{"action": "existing"|"new", "chunk_id": int|null, "title": str, "summary": str, "keywords": [str]}`.
Placement and metadata refresh happen in that one call. A chunk that already holds
`max_props` propositions is closed (not offered as a candidate). Sections are
independent and processed in parallel; their chunks are concatenated in document order.

- [ ] **Step 1: Create the feature branch**

```bash
git checkout main && git pull --ff-only && git checkout -b feat/agent
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_agent.py`:

```python
from agentic_chunker._common import Proposition
from agentic_chunker._agent import assign


def P(text, header, start=0, end=10):
    return Proposition(text=text, char_start=start, char_end=end, header=header)


def test_new_then_existing_builds_one_chunk():
    props = [P("Cats purr.", "Cats", 0, 10), P("Cats sleep a lot.", "Cats", 0, 10)]

    def fake_decide(prop_text, open_chunks, cfg):
        if not open_chunks:
            return {"action": "new", "title": "Cats", "summary": "About cats.", "keywords": ["cats"]}
        return {"action": "existing", "chunk_id": open_chunks[0]["id"],
                "title": "Cats", "summary": "About cats and sleep.", "keywords": ["cats", "sleep"]}

    chunks = assign(props, cfg=None, decide=fake_decide)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.index == 0
    assert c.text == "Cats purr.\nCats sleep a lot."
    assert c.title == "Cats"
    assert c.summary == "About cats and sleep."
    assert c.keywords == ["cats", "sleep"]


def test_action_new_starts_second_chunk():
    props = [P("Cats purr.", "Sec", 0, 10), P("Dogs bark.", "Sec", 0, 10)]

    def fake_decide(prop_text, open_chunks, cfg):
        return {"action": "new", "title": prop_text, "summary": prop_text, "keywords": []}

    chunks = assign(props, cfg=None, decide=fake_decide)
    assert [c.text for c in chunks] == ["Cats purr.", "Dogs bark."]
    assert [c.index for c in chunks] == [0, 1]


def test_sections_never_merge():
    props = [P("Alpha.", "A", 0, 6), P("Beta.", "B", 10, 15)]

    def fake_decide(prop_text, open_chunks, cfg):
        # Always try to reuse an existing chunk; cross-section reuse must be impossible.
        if open_chunks:
            return {"action": "existing", "chunk_id": open_chunks[0]["id"],
                    "title": "x", "summary": "x", "keywords": []}
        return {"action": "new", "title": "x", "summary": "x", "keywords": []}

    chunks = assign(props, cfg=None, decide=fake_decide)
    assert len(chunks) == 2  # one per section, never merged


def test_max_props_forces_new_chunk():
    props = [P(f"Fact {i}.", "Sec", 0, 5) for i in range(3)]

    def fake_decide(prop_text, open_chunks, cfg):
        # Greedily wants to add to an existing chunk every time.
        if open_chunks:
            return {"action": "existing", "chunk_id": open_chunks[0]["id"],
                    "title": "t", "summary": "s", "keywords": []}
        return {"action": "new", "title": "t", "summary": "s", "keywords": []}

    chunks = assign(props, cfg=None, decide=fake_decide, max_props=2)
    # 3 props, cap 2 -> chunk0 gets 2, chunk1 gets 1
    assert [len(c.text.split("\n")) for c in chunks] == [2, 1]


def test_source_spans_aggregated_and_deduped():
    props = [P("a", "Sec", 0, 5), P("b", "Sec", 0, 5), P("c", "Sec", 8, 12)]

    def fake_decide(prop_text, open_chunks, cfg):
        if open_chunks:
            return {"action": "existing", "chunk_id": open_chunks[0]["id"],
                    "title": "t", "summary": "s", "keywords": []}
        return {"action": "new", "title": "t", "summary": "s", "keywords": []}

    chunks = assign(props, cfg=None, decide=fake_decide)
    assert chunks[0].source_spans == [(0, 5), (8, 12)]


def test_decide_failure_falls_back_to_new_chunk():
    props = [P("a", "Sec", 0, 5), P("b", "Sec", 0, 5)]

    def fake_decide(prop_text, open_chunks, cfg):
        return None  # simulate LLM/parse failure

    chunks = assign(props, cfg=None, decide=fake_decide)
    # every failure -> its own new chunk
    assert [c.text for c in chunks] == ["a", "b"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentic_chunker._agent'`

- [ ] **Step 4: Write minimal implementation**

Create `src/agentic_chunker/_agent.py`:

```python
"""Per-proposition placement agent loop.

Groups propositions within each header (section) independently and in parallel.
For each proposition, `decide()` asks the LLM whether it joins an open chunk or
starts a new one, and returns refreshed title/summary/keywords in the same call.
Chunks at capacity are closed. On any decide failure the proposition starts a new
chunk so content is never dropped.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from ._common import Chunk, Proposition
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


def _default_decide(prop_text: str, open_chunks: list[dict], cfg):
    import json
    payload = chat_json = _real_chat_json(
        _PROMPT.format(open_chunks=json.dumps(open_chunks, ensure_ascii=False), prop=prop_text),
        cfg,
    )
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
```

> Note on the `_default_decide` line `payload = chat_json = _real_chat_json(...)`: simplify to `payload = _real_chat_json(...)` when implementing — the double-assignment is a typo to remove. (Listed here so the engineer fixes it rather than copying blindly.)

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_agent.py -v`
Expected: PASS (6 passed)

- [ ] **Step 6: Commit, push, open PR**

```bash
git add src/agentic_chunker/_agent.py tests/test_agent.py
git commit -m "feat: add per-proposition placement agent loop"
git push -u origin feat/agent
gh pr create --fill --base main
gh pr merge --merge --delete-branch
```

---

## Task 6: Public API wiring (`__init__.py`) + end-to-end test

**Branch:** `feat/public-api`

**Files:**
- Modify: `src/agentic_chunker/__init__.py`
- Test: `tests/test_chunker.py`

- [ ] **Step 1: Create the feature branch**

```bash
git checkout main && git pull --ff-only && git checkout -b feat/public-api
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_chunker.py`. It monkeypatches the stage-level callables the
`AgenticChunker` uses, so it exercises real wiring without a live LLM.

```python
import agentic_chunker as ac
from agentic_chunker import AgenticChunker, LlmConfig, Chunk


CFG = LlmConfig(url="http://x/v1", api_key="k", model="m")


def test_chunk_end_to_end_wiring(monkeypatch):
    # Stub extraction: one proposition per block (block text passthrough).
    def fake_extract(blocks, cfg, **kw):
        from agentic_chunker._common import Proposition
        return [Proposition(b.text, b.char_start, b.char_end, b.header) for b in blocks]

    # Stub assign: one chunk per proposition, echoing text.
    def fake_assign(props, cfg, **kw):
        return [Chunk(index=i, text=p.text, title=p.text[:5], summary=p.text,
                      keywords=[], source_spans=[(p.char_start, p.char_end)])
                for i, p in enumerate(props)]

    monkeypatch.setattr(ac, "_extract", fake_extract)
    monkeypatch.setattr(ac, "_assign", fake_assign)

    md = "# H\n\nAlpha para.\n\nBeta para."
    chunker = AgenticChunker(llm=CFG)
    chunks = chunker.chunk(md)

    assert [c.text for c in chunks] == ["Alpha para.", "Beta para."]
    assert all(isinstance(c, Chunk) for c in chunks)
    assert [c.index for c in chunks] == [0, 1]


def test_empty_input_returns_empty_list():
    chunker = AgenticChunker(llm=CFG)
    assert chunker.chunk("") == []


def test_exports_available():
    assert hasattr(ac, "AgenticChunker")
    assert hasattr(ac, "LlmConfig")
    assert hasattr(ac, "Chunk")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_chunker.py -v`
Expected: FAIL with `ImportError: cannot import name 'AgenticChunker'` (or AttributeError on `chunk`).

- [ ] **Step 4: Write minimal implementation**

Replace `src/agentic_chunker/__init__.py` with:

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_chunker.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest -v`
Expected: all tests pass across every test file.

- [ ] **Step 7: Commit, push, open PR**

```bash
git add src/agentic_chunker/__init__.py tests/test_chunker.py
git commit -m "feat: wire AgenticChunker public API end-to-end"
git push -u origin feat/public-api
gh pr create --fill --base main
gh pr merge --merge --delete-branch
```

---

## Self-Review notes (for the planner; not execution steps)

- **Spec coverage:** input=text (Task 6 `chunk(str)`); proposition strategy (Tasks 4+5); header pre-split + parallel extraction (Tasks 3+4); section-bounded placement + parallel sections (Task 5); combined placement/metadata call (Task 5 `decide`); output fields text/title/summary/keywords/source_spans/index (Tasks 1,5); approximate source spans via block offsets (Tasks 3,4,5); LLM via urllib OpenAI-compatible (Task 2); error fallbacks for extraction (Task 4) and placement (Task 5); stdlib-only (all tasks); stub-LLM deterministic tests (Tasks 2,4,5,6). All spec sections map to a task.
- **Known typo deliberately flagged:** the `_default_decide` double-assignment in Task 5 Step 4 — the note instructs the engineer to fix it.
- **Type consistency:** `Block`/`Proposition`/`Chunk` fields used in Tasks 3-6 match Task 1 definitions; `extract(blocks, cfg, *, chat_json, concurrency)` and `assign(props, cfg, *, decide, max_props, concurrency)` signatures used consistently in `__init__.py` (positional `cfg`, keyword `concurrency`/`max_props`).
