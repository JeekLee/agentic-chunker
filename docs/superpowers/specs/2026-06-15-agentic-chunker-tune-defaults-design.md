# agentic-chunker v2.2 — Tune Defaults for Local Models

**Date:** 2026-06-15
**Status:** Approved (design); implementation pending
**Builds on:** the v2 batch design + v2.1 dedupe design.

## Motivation

Dumping the full chunks of `03_diagram_hwp.md` through the **public `AgenticChunker`
with default settings** produced 163 fail-soft chunks (one per proposition) instead of
the expected ~30 — every `group` LLM call timed out. The defaults (`LlmConfig.timeout=60`,
`AgenticChunker.max_concurrency=8`) are too aggressive for a single self-hosted model:
8 concurrent large grouping prompts queue on one backend and each exceeds the 60 s
timeout, so the pipeline silently degrades to its fail-soft path.

The same document with `timeout=120` + `max_concurrency=4` completed cleanly (0 timeouts,
33 chunks, ~1.8 min). Since this project primarily targets a local OpenAI-compatible
endpoint (the md-converter pattern), the defaults should match what reliably works there.

## Decisions

| Setting | Old default | New default |
|---------|-------------|-------------|
| `LlmConfig.timeout` | 60 | **120** |
| `AgenticChunker.max_concurrency` | 8 | **4** |

These are conservative and safe for fast cloud APIs too (timeout is only a ceiling;
concurrency 4 is ample). README gains a short tuning note. No logic changes.

## Affected files

- `src/agentic_chunker/llm.py` — `LlmConfig.timeout` default `60` → `120`.
- `src/agentic_chunker/__init__.py` — `AgenticChunker.__init__` `max_concurrency` default `8` → `4` (docstring unchanged in wording).
- `README.md` — add a "Tuning for local models" note.
- `tests/test_llm.py`, `tests/test_chunker.py` — add default-pinning tests.
- `_agent.py`, `_propositions.py`, `_common.py`, `_split.py` — unchanged.

## Changes

### Defaults
- `llm.py`: `timeout: int = 120` (was 60).
- `__init__.py`: `max_concurrency: int = 4` (was 8). `window_size` (40) and
  `min_extract_chars` (20) are unchanged.

### README tuning note
Add a short subsection after Usage, e.g.:

> ### Tuning for your LLM endpoint
> Agentic chunking makes many LLM calls. Defaults (`timeout=120`, `max_concurrency=4`)
> suit a single self-hosted model. For a fast hosted API you can raise `max_concurrency`
> for throughput; for a large/slow endpoint raise `LlmConfig(timeout=...)`. If you see
> many `chat failed: timed out` lines on stderr, the pipeline is falling back to one
> chunk per proposition — raise the timeout and/or lower concurrency.

### Tests (regression-pin the defaults)
- `tests/test_llm.py`: assert `LlmConfig(url=..., api_key=..., model=...).timeout == 120`.
- `tests/test_chunker.py`: construct `AgenticChunker(llm=CFG)` with only `llm`, monkeypatch
  `_extract`/`_assign`, and assert the forwarded `concurrency == 4` (and `min_extract_chars == 20`
  default still holds).

## Out of scope (YAGNI)

- Auto-tuning concurrency/timeout from endpoint probing.
- Changing `window_size` / `min_extract_chars` / `max_propositions_per_chunk` defaults.
- Retry logic on timeout (fail-soft already covers it).
