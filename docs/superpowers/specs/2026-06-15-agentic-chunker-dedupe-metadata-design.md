# agentic-chunker v2.1 — De-duplicate Chunk Metadata Design

**Date:** 2026-06-15
**Status:** Approved (design); implementation pending
**Builds on:** [`2026-06-15-agentic-chunker-batch-v2-design.md`](2026-06-15-agentic-chunker-batch-v2-design.md)

## Motivation

The v2 batch benchmark on `03_diagram_hwp.md` hit the performance/hallucination goals
(~29 min → ~1.5 min; bare labels emitted verbatim) but surfaced a quality issue:
**consecutive chunks with identical `title` / `summary` / `keywords`.**

Root cause: the `group` LLM call sometimes merges many propositions into one large
cluster (> `max_props`). The v2 post-cap splits such a cluster into consecutive
sub-chunks, and each sub-chunk **copies the cluster's title/summary/keywords verbatim** —
so the metadata no longer distinguishes them. (Not a correctness bug — chunk *text*
differs and no content is lost — but the metadata is non-discriminating for retrieval.)

## Decisions

| Topic | Decision |
|-------|----------|
| Reduce over-merging (root cause) | Pass `max_props` into the `group` prompt so the model targets cohesive clusters of ≤ `max_props` propositions and splits long topics itself. |
| Distinguish forced splits (safety net) | When the post-cap splits a cluster into N > 1 sub-chunks, suffix each sub-chunk's `title` with `" (k/N)"`. N == 1 → no marker. |
| Hard cap | **Kept.** The model may not honor the prompt target, so the post-cap remains the safety pin (now producing distinguishable titles). |
| `max_props` meaning | Unchanged: a per-chunk **proposition count** cap (default 10), grounded in the *Dense X Retrieval* ~100–200 word / ~10 proposition retrieval sweet spot. A coarse size proxy; switching to a token/char budget is explicitly out of scope. |

## Affected files

- `src/agentic_chunker/_agent.py` — `_PROMPT`, `_default_group`, `_group_window` (and the `group` seam signature).
- `tests/test_agent.py` — update fakes to the new seam signature; update the post-cap test; add marker/guidance tests.
- `_propositions.py`, `__init__.py`, `_common.py`, `_split.py`, `llm.py` — unchanged. `max_props` already flows from the public API to `assign`.

## Changes

### 1. `group` seam gains `max_props`

The injected grouping callable changes from `group(prop_texts, cfg)` to
**`group(prop_texts, cfg, max_props) -> list[dict] | None`**. `_group_window` passes the
effective `max_props` through. This lets the default grouper put the size target into the
prompt, and keeps the seam explicit for tests.

### 2. `_default_group` prompt embeds the target

`_PROMPT` gains a `{max_props}` placeholder and instruction text telling the model to keep
each chunk to about `max_props` propositions, split long topics, and separate distinct
topics. Prompt assembly uses `.replace("{max_props}", str(max_props)).replace("{props}", numbered)`
— two single-token replacements, leaving the literal `{` / `}` of the JSON example intact
(same brace-safety rationale as v2). Returns the payload only if it is a list, else `None`.

### 3. Part marker on forced splits (`_group_window`)

When a valid cluster's members exceed `max_props`, it is split into consecutive sub-chunks
of ≤ `capped = max(1, max_props)` (unchanged). New: if the split yields `N > 1` sub-chunks,
each sub-chunk's `title` becomes `f"{title} ({k}/{N})"` for `k` in `1..N`. A single-part
cluster (`N == 1`) keeps the title unchanged. `summary` and `keywords` are unchanged across
parts (the title marker is sufficient to disambiguate). The `_own_chunks` fallback is
untouched — those chunks already have distinct titles (`text[:40]`).

## Invariants preserved

Fail-soft (group failure / non-list / empty result → one chunk per proposition); window
partitioning + parallelism; index validation (out-of-range / duplicate dropped);
unassigned-proposition → own chunk; `source_spans` dedup; sequential `index`; `Chunk` schema.

## Testing (TDD, stub LLM)

- Update all existing `_agent` test fakes to the `group(texts, cfg, max_props)` signature.
- `test_max_props_post_cap_splits_oversized_cluster`: expect titles `"t (1/3)"`, `"t (2/3)"`,
  `"t (3/3)"` for a 5-member cluster at `max_props=2`.
- New: a cluster within the cap (`N == 1`) keeps its title unchanged (no marker).
- New: `_group_window` passes the effective `max_props` to `group` (assert the fake received it).
- New: marker format is exactly `"{title} ({k}/{N})"` (e.g. verify the second part of a
  3-way split is `"t (2/3)"`).
- Existing window/section/fail-soft/index/span tests keep passing.

## Verification

Re-run the benchmark on `03_diagram_hwp.md`. Expect: no consecutive chunks with identical
titles (forced splits now carry `(k/N)`; ideally fewer splits because the prompt curbs
over-merging), with total runtime still ~v2 levels (~1.5 min). Manual check via the
`examples/smoke_test.py`-style timing script; the automated suite stays offline with a stub LLM.

## Out of scope (YAGNI)

- Re-summarizing each sub-chunk with an extra LLM call (rejected — adds calls).
- Cross-window de-duplication / hierarchical merge.
- Token/char-budget chunk sizing (still proposition-count based).
