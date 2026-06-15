# agentic-chunker v2 — Batch Chunking Design

**Date:** 2026-06-15
**Status:** Approved (design); implementation pending
**Supersedes:** the placement mechanism in [`2026-06-15-agentic-chunker-design.md`](2026-06-15-agentic-chunker-design.md) (extraction/split/output model unchanged)

## Motivation

A real-document benchmark exposed two problems with the v1 per-proposition agent loop:

| Doc | chars | blocks | props | chunks | extract | assign | total |
|-----|-------|--------|-------|--------|---------|--------|-------|
| `03_diagram_hwp.md` | 2,750 | 86 | 264 | 46 | 96s | **1,639s** | **~29 min** |

1. **Placement is the bottleneck (94% of runtime).** v1 makes one LLM call per
   proposition, sequentially within a section. Documents with no `#` headers form a
   single section, so all 264 calls run serially (~6.2s each). This does not scale and
   cannot be fixed by tuning timeout/concurrency — it is structural.
2. **Short labels hallucinate.** Diagram/cover content yields bare-label blocks
   ("목적", "보건복지부"). v1 sends each to the extractor, which invents generic
   encyclopedia-style propositions not present in the source.

(With tuned settings — `timeout=120`, `concurrency=4` — the run completed with 0
timeouts and 0 parse failures, so the pipeline is robust; it is just impractically slow
and low-quality on non-prose input.)

## Decisions

| Topic | Decision |
|-------|----------|
| Placement strategy | **Replace** per-proposition loop with **batch grouping**: one LLM call clusters a whole section's propositions; sections larger than `window_size` are split into contiguous windows, each clustered independently. |
| v1 per-proposition loop | **Removed** (recoverable from git history if ever needed). |
| Hallucination fix | **Skip extraction for short blocks**: blocks shorter than `min_extract_chars` become a single verbatim proposition with no LLM call. (Prompt-guard approach not adopted.) |
| Cross-window merging | **Not done** (no hierarchical second pass). Windows are contiguous document-order slices, so adjacent related propositions usually co-locate. Accepted tradeoff. |
| `max_propositions_per_chunk` | Kept as a **post-processing hard cap**: clusters exceeding it are split. |

## Affected files

- `src/agentic_chunker/_propositions.py` — add `min_extract_chars` short-block bypass.
- `src/agentic_chunker/_agent.py` — **rewrite**: batch grouping replaces the per-proposition loop. `assign` keeps its name; injected seam becomes `group` (was `decide`).
- `src/agentic_chunker/__init__.py` — new constructor params (`window_size`, `min_extract_chars`); drop the per-proposition path.
- Tests: `tests/test_propositions.py`, `tests/test_agent.py`, `tests/test_chunker.py` updated.
- `_common.py`, `_split.py`, `llm.py` — unchanged.

## Public API

```python
AgenticChunker(
    *,
    llm: LlmConfig,
    max_propositions_per_chunk: int = 10,   # post-cap: clusters larger than this are split
    window_size: int = 40,                  # props per grouping call; larger sections are windowed
    min_extract_chars: int = 20,            # blocks shorter than this skip LLM extraction
    max_concurrency: int = 8,
)
chunks: list[Chunk] = chunker.chunk(markdown_text)
```

`Chunk` schema is unchanged: `index`, `text`, `title`, `summary`, `keywords`, `source_spans`.

## Stage 1 — extraction change (`_propositions.extract`)

New signature:
```python
def extract(blocks, cfg, *, chat_json=_real_chat_json, concurrency=8, min_extract_chars=20) -> list[Proposition]
```
For each block: if `len(block.text) < min_extract_chars`, emit a single `Proposition`
with `text = block.text` (verbatim) and the block's span/header — **no LLM call**.
Otherwise behave as today (parallel LLM extraction, fail-soft to whole-block proposition).

## Stage 2 — batch grouping (`_agent`)

New signature:
```python
def assign(props, cfg, *, group=_default_group, max_props=10, window_size=40, concurrency=8) -> list[Chunk]
```

Algorithm:
1. Partition `props` by `header` (section), preserving first-seen section order (unchanged from v1).
2. For each section, split its propositions into **contiguous windows** of at most
   `window_size` (a section with ≤ `window_size` props is a single window). Build a flat
   ordered list of windows, each tagged with its `(section_order, window_order)`.
3. For each window, call `group(prop_texts, cfg)` — one LLM call. Windows run in parallel
   via `ThreadPoolExecutor(max_workers=max(1, concurrency))`.
4. `group` returns clusters: `[{"proposition_indices": [int], "title": str, "summary": str, "keywords": [str]}, ...]`
   where indices refer to positions in that window's `prop_texts`.
5. Build chunks per window from its clusters (see validation/fallback below). Concatenate
   windows in `(section_order, window_order)` order; concatenate clusters in returned order.
6. Assign sequential `index` across all chunks. Each chunk: `text` = member propositions
   joined by `"\n"`; `source_spans` = deduped `(char_start, char_end)` per member in
   first-seen order; `title`/`summary`/`keywords` from the cluster.

### `_default_group` prompt
One user message embedding the indexed proposition list (concatenation, **not** `str.format`,
to stay brace-safe like v1 extraction). Asks the model to group indices into semantically
coherent chunks and return ONLY the JSON array described above. Korean prompt, matching the
project's existing style.

### Index validation, capacity cap, and fallback
- Drop indices that are out of range or duplicated (first occurrence wins).
- Any proposition not assigned to any cluster → its own single-proposition chunk.
- A cluster with more than `max_props` members is split into consecutive sub-chunks of
  at most `max_props` (preserving order); the cluster's title/summary/keywords apply to
  each resulting sub-chunk.
- If `group` returns `None` / not a list / no valid clusters → **fail-soft**: every
  proposition in the window becomes its own chunk (`title = text[:40]`, `summary = text`,
  `keywords = []`). Content is never lost.

## Error handling

Mirrors v1: every LLM-touching path is fail-soft (stderr log + safe fallback). Extraction
failure → whole-block proposition; grouping failure → one chunk per proposition. The
pipeline never raises on LLM/parse errors.

## Testing (TDD, stub LLM)

- `_propositions`: short block (`< min_extract_chars`) skips extraction and is emitted
  verbatim with no `chat_json` call; boundary case at exactly the threshold; existing
  extraction behavior preserved for longer blocks.
- `_agent`: single-call clustering of a small section; window splitting for a section
  larger than `window_size` (verify number of `group` calls and that windows are contiguous
  slices); `max_props` post-cap splits oversized clusters; index validation (out-of-range /
  duplicate dropped); unassigned proposition → own chunk; `group` failure → one chunk per
  proposition; source-span aggregation/dedup; section + window ordering preserved.
- `__init__`: new params forwarded to extract/assign; end-to-end wiring with stubbed
  `_extract`/`_assign` (as in v1's test).

## Verification

Re-run the benchmark on `03_diagram_hwp.md` after implementation. Target: total runtime
from ~29 min down to the low single-minutes or less, with bare-label blocks no longer
hallucinated (emitted verbatim). This is a manual check via `examples/smoke_test.py`-style
timing; the automated suite stays offline with a stub LLM.

## Out of scope (YAGNI)

- Hierarchical cross-window merge pass.
- Keeping the v1 per-proposition loop as a selectable strategy.
- Embedding/vector-store integration.
- Special table parsing (tables are still treated as text blocks).
