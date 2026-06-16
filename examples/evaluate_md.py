"""Evaluate Markdown files across speed, chunking, graph, and search signals.

Run with a real OpenAI-compatible endpoint:

    LLM_URL=http://172.18.0.3:8000/v1 \
    LLM_API_KEY= \
    LLM_MODEL=qwen3-vl-30b-a3b \
    .venv/bin/python examples/evaluate_md.py /tmp/mdout/val_01_image_hwpx.md

Run deterministic parsing/graph checks without LLM calls:

    .venv/bin/python examples/evaluate_md.py /tmp/mdout/*.md --no-llm
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import statistics
import time
from typing import Any

from agentic_chunker import AgenticChunker, ChunkingResult, LlmConfig
from agentic_chunker._graph import attach_document_graphs
from agentic_chunker._grouping import group_units
import agentic_chunker._grouping as grouping_mod
from agentic_chunker._references import table_references
from agentic_chunker._split import split
from agentic_chunker._units import build_units


def main() -> int:
    args = _parse_args()
    cfg = None if args.no_llm else _llm_config(args)
    reports = [_evaluate_path(path, cfg, args) for path in args.paths]
    if args.aggregate_only:
        payload = _aggregate_reports(reports)
    elif len(reports) == 1:
        payload = reports[0]
    else:
        payload = {
            "aggregate": _aggregate_reports(reports),
            "files": reports,
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--llm-url", default=os.environ.get("LLM_URL", ""))
    parser.add_argument("--llm-api-key", default=os.environ.get("LLM_API_KEY", ""))
    parser.add_argument("--llm-model", default=os.environ.get("LLM_MODEL", ""))
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("LLM_TIMEOUT", "180")))
    parser.add_argument("--max-units", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--max-good-source-chars", type=int, default=6000)
    parser.add_argument(
        "--gold-query",
        action="append",
        default=[],
        metavar="QUERY::EXPECTED_TEXT",
        help="Optional lexical retrieval check. EXPECTED_TEXT may be omitted.",
    )
    return parser.parse_args()


def _llm_config(args: argparse.Namespace) -> LlmConfig:
    if not args.llm_url or not args.llm_model:
        raise SystemExit("Set LLM_URL and LLM_MODEL, or pass --no-llm.")
    return LlmConfig(
        url=args.llm_url,
        api_key=args.llm_api_key,
        model=args.llm_model,
        timeout=args.timeout,
    )


def _run(
    markdown: str,
    cfg: LlmConfig | None,
    args: argparse.Namespace,
) -> tuple[ChunkingResult, list[dict[str, Any]], float]:
    llm_calls: list[dict[str, Any]] = []
    original_chat_json = grouping_mod._real_chat_json

    def timed_chat_json(prompt: str, llm_cfg: LlmConfig) -> object | None:
        started = time.perf_counter()
        result = original_chat_json(prompt, llm_cfg)
        llm_calls.append({
            "kind": _call_kind(prompt),
            "prompt_chars": len(prompt),
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "ok": result is not None,
            "result_type": type(result).__name__,
        })
        return result

    started = time.perf_counter()
    if cfg is None:
        blocks = split(markdown)
        units = build_units(blocks)
        chunks = group_units(
            units,
            cfg=None,
            group=lambda _units, _cfg, _max_units: None,
            max_units=args.max_units,
            window_size=args.window_size,
            concurrency=1,
        )
        document_graph = attach_document_graphs(chunks)
        result = ChunkingResult(chunks=chunks, document_graph=document_graph)
    else:
        grouping_mod._real_chat_json = timed_chat_json
        try:
            chunker = AgenticChunker(
                llm=cfg,
                max_units_per_chunk=args.max_units,
                window_size=args.window_size,
                max_concurrency=args.max_concurrency,
                document_graph=True,
            )
            result = chunker.chunk_document(markdown)
        finally:
            grouping_mod._real_chat_json = original_chat_json
    wall_sec = time.perf_counter() - started
    return result, llm_calls, wall_sec


def _evaluate_path(path: Path, cfg: LlmConfig | None, args: argparse.Namespace) -> dict[str, Any]:
    markdown = path.read_text(encoding=args.encoding)
    result, llm_calls, wall_sec = _run(markdown, cfg, args)
    blocks = split(markdown)
    units = build_units(blocks)
    return _report(
        path=path,
        markdown=markdown,
        blocks=blocks,
        units=units,
        result=result,
        llm_calls=llm_calls,
        wall_sec=wall_sec,
        args=args,
        cfg=cfg,
    )


def _call_kind(prompt: str) -> str:
    if "evidence unit 목록" in prompt:
        return "group"
    if "청크 메타데이터 작성자" in prompt:
        return "enrich"
    if "질문을 정확히 3개" in prompt:
        return "retry_questions"
    return "other"


def _aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "files": len(reports),
        "input": {
            "bytes": sum(report["input"]["bytes"] for report in reports),
            "chars": sum(report["input"]["chars"] for report in reports),
            "lines": sum(report["input"]["lines"] for report in reports),
            "blocks": sum(report["input"]["blocks"] for report in reports),
            "units": sum(report["input"]["units"] for report in reports),
            "unit_kinds": _sum_counters(report["input"]["unit_kinds"] for report in reports),
        },
        "speed": {
            "wall_sec": round(sum(report["speed"]["wall_sec"] for report in reports), 3),
            "llm_calls": sum(report["speed"]["llm_calls"] for report in reports),
        },
        "chunking_quality": {
            "chunks": sum(report["chunking_quality"]["chunks"] for report in reports),
            "tiny_chunks": sum(report["chunking_quality"]["tiny_chunks"] for report in reports),
            "oversized_chunks": sum(report["chunking_quality"]["oversized_chunks"] for report in reports),
            "unit_coverage_ok": all(
                not report["chunking_quality"]["unit_coverage"]["missing"]
                and not report["chunking_quality"]["unit_coverage"]["duplicates"]
                for report in reports
            ),
        },
        "graph_quality": {
            "nodes": sum(report["graph_quality"]["nodes"] for report in reports),
            "edges": sum(report["graph_quality"]["edges"] for report in reports),
            "edge_types": _sum_counters(report["graph_quality"]["edge_types"] for report in reports),
            "avg_table_reference_coverage": _average(
                report["graph_quality"]["table_reference_coverage"] for report in reports
            ),
            "files_with_missing_table_refs": sum(
                1 for report in reports if report["graph_quality"]["missing_table_reference_edges"]
            ),
        },
        "search_quality_expected": {
            "avg_metadata_complete_ratio": _average(
                report["search_quality_expected"]["metadata_complete_ratio"] for report in reports
            ),
            "chunks_missing_keywords": sum(
                report["search_quality_expected"]["chunks_missing_keywords"] for report in reports
            ),
            "chunks_with_questions_lt_2": sum(
                report["search_quality_expected"]["chunks_with_questions_lt_2"] for report in reports
            ),
            "avg_table_context_coverage": _average(
                report["search_quality_expected"]["table_context_coverage"] for report in reports
            ),
            "gold_hit_at_5": _aggregate_gold_hit(reports),
        },
    }


def _sum_counters(items: Any) -> dict[str, int]:
    counter: Counter = Counter()
    for item in items:
        counter.update(item)
    return dict(counter)


def _average(values: Any) -> float:
    values = list(values)
    return round(statistics.mean(values), 4) if values else 0.0


def _aggregate_gold_hit(reports: list[dict[str, Any]]) -> float | None:
    values = [
        report["search_quality_expected"]["lexical_gold_queries"]["hit_at_5"]
        for report in reports
        if report["search_quality_expected"]["lexical_gold_queries"]["hit_at_5"] is not None
    ]
    return _average(values) if values else None


def _report(
    *,
    path: Path,
    markdown: str,
    blocks: list,
    units: list,
    result: ChunkingResult,
    llm_calls: list[dict[str, Any]],
    wall_sec: float,
    args: argparse.Namespace,
    cfg: LlmConfig | None,
) -> dict[str, Any]:
    chunks = result.chunks
    expected_unit_ids = {unit.index for unit in units}
    assigned_unit_ids = [
        item["unit_index"]
        for chunk in chunks
        for item in chunk.metadata.get("units", [])
        if isinstance(item.get("unit_index"), int)
    ]
    assigned_counts = Counter(assigned_unit_ids)
    raw_refs = set(table_references(markdown))
    ref_ids = {
        edge.metadata.get("table_id")
        for edge in result.document_graph.edges
        if edge.type == "REFERS_TO" and edge.metadata.get("table_id")
    }
    chunk_lengths = [len(chunk.source) for chunk in chunks] or [0]
    units_per_chunk = [len(chunk.metadata.get("units", [])) for chunk in chunks] or [0]

    return {
        "input": {
            "path": str(path),
            "bytes": path.stat().st_size,
            "chars": len(markdown),
            "lines": markdown.count("\n") + 1,
            "blocks": len(blocks),
            "units": len(units),
            "unit_kinds": dict(Counter(_unit_kind(unit) for unit in units)),
        },
        "config": {
            "mode": "deterministic" if cfg is None else "llm",
            "model": cfg.model if cfg else None,
            "max_units": args.max_units,
            "window_size": args.window_size,
            "max_concurrency": args.max_concurrency if cfg else 1,
        },
        "speed": {
            "wall_sec": round(wall_sec, 3),
            "llm_calls": len(llm_calls),
            "llm_call_summary": _llm_call_summary(llm_calls),
        },
        "chunking_quality": {
            "chunks": len(chunks),
            "unit_coverage": {
                "assigned": len(assigned_unit_ids),
                "expected": len(expected_unit_ids),
                "missing": sorted(expected_unit_ids - set(assigned_unit_ids)),
                "duplicates": sorted(unit_id for unit_id, count in assigned_counts.items() if count > 1),
            },
            "source_chars": _distribution(chunk_lengths),
            "units_per_chunk": _distribution(units_per_chunk),
            "tiny_chunks": sum(1 for chunk in chunks if len(chunk.source.strip()) < 20),
            "oversized_chunks": sum(1 for chunk in chunks if len(chunk.source) > args.max_good_source_chars),
        },
        "graph_quality": {
            "nodes": len(result.document_graph.nodes),
            "edges": len(result.document_graph.edges),
            "node_types": dict(Counter(node.type for node in result.document_graph.nodes)),
            "edge_types": dict(Counter(edge.type for edge in result.document_graph.edges)),
            "table_reference_coverage": _ratio(len(raw_refs & ref_ids), len(raw_refs)),
            "missing_table_reference_edges": sorted(raw_refs - ref_ids),
        },
        "search_quality_expected": {
            "metadata_complete_ratio": _ratio(
                sum(1 for chunk in chunks if chunk.summary and chunk.keywords and len(chunk.questions_answered) >= 2),
                len(chunks),
            ),
            "chunks_missing_keywords": sum(1 for chunk in chunks if not chunk.keywords),
            "chunks_with_questions_lt_2": sum(1 for chunk in chunks if len(chunk.questions_answered) < 2),
            "avg_embedding_chars": round(statistics.mean(len(chunk.embedding_text) for chunk in chunks), 1) if chunks else 0,
            "table_context_coverage": _ratio(len(raw_refs & ref_ids), len(raw_refs)),
            "lexical_gold_queries": _lexical_gold_report(chunks, args.gold_query),
        },
    }


def _llm_call_summary(calls: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary = {}
    for kind in sorted({call["kind"] for call in calls}):
        rows = [call for call in calls if call["kind"] == kind]
        summary[kind] = {
            "count": len(rows),
            "ok": sum(1 for row in rows if row["ok"]),
            "failed": sum(1 for row in rows if not row["ok"]),
            "avg_sec": round(statistics.mean(row["elapsed_sec"] for row in rows), 3),
            "max_sec": round(max(row["elapsed_sec"] for row in rows), 3),
            "avg_prompt_chars": round(statistics.mean(row["prompt_chars"] for row in rows)),
            "max_prompt_chars": max(row["prompt_chars"] for row in rows),
        }
    return summary


def _distribution(values: list[int]) -> dict[str, float | int]:
    return {
        "min": min(values),
        "median": round(statistics.median(values), 1),
        "avg": round(statistics.mean(values), 1),
        "max": max(values),
    }


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0
    return round(numerator / denominator, 4)


def _unit_kind(unit: Any) -> str:
    return unit.metadata.get("common", {}).get("chunk_kind", "?")


def _lexical_gold_report(chunks: list, specs: list[str], top_k: int = 5) -> dict[str, Any]:
    queries = [_parse_gold_query(spec) for spec in specs]
    if not queries:
        return {"count": 0, "hit_at_5": None, "queries": []}

    results = []
    hits = 0
    for query, expected in queries:
        ranked = _rank_chunks(query, chunks)[:top_k]
        hit = bool(expected) and any(_contains_expected(chunk, expected) for _, chunk in ranked)
        if hit:
            hits += 1
        results.append({
            "query": query,
            "expected": expected,
            "hit_at_5": hit if expected else None,
            "top_chunks": [
                {
                    "index": chunk.index,
                    "score": score,
                    "title": chunk.title,
                    "preview": chunk.source.replace("\n", " ")[:160],
                }
                for score, chunk in ranked
            ],
        })
    expected_count = sum(1 for _, expected in queries if expected)
    return {
        "count": len(queries),
        "hit_at_5": _ratio(hits, expected_count) if expected_count else None,
        "queries": results,
    }


def _parse_gold_query(spec: str) -> tuple[str, str]:
    if "::" not in spec:
        return spec.strip(), ""
    query, expected = spec.split("::", 1)
    return query.strip(), expected.strip()


def _rank_chunks(query: str, chunks: list) -> list[tuple[int, Any]]:
    query_terms = _terms(query)
    ranked = []
    for chunk in chunks:
        text = _search_text(chunk)
        terms = _terms(text)
        term_counts = Counter(terms)
        score = sum(term_counts.get(term, 0) for term in query_terms)
        if query and query in text:
            score += 5
        for table_id in _chunk_table_ids(chunk):
            if table_id and table_id in query:
                score += 80
        if chunk.title and chunk.title in query:
            score += 30
        ranked.append((score, chunk))
    return sorted(ranked, key=lambda item: (-item[0], item[1].index))


def _search_text(chunk: Any) -> str:
    parts = [
        chunk.title,
        chunk.summary,
        " ".join(chunk.keywords),
        " ".join(chunk.questions_answered),
        chunk.embedding_text,
        chunk.source,
    ]
    return "\n".join(part for part in parts if part)


def _terms(text: str) -> list[str]:
    return [term.lower() for term in re.findall(r"[0-9A-Za-z가-힣]+", text)]


def _contains_expected(chunk: Any, expected: str) -> bool:
    needle = expected.lower()
    return needle in _search_text(chunk).lower()


def _chunk_table_ids(chunk: Any) -> list[str]:
    table_ids: list[str] = []
    table = chunk.metadata.get("table") if isinstance(chunk.metadata, dict) else None
    if isinstance(table, dict) and isinstance(table.get("table_id"), str):
        table_ids.append(table["table_id"])
    tables = chunk.metadata.get("tables") if isinstance(chunk.metadata, dict) else None
    if isinstance(tables, list):
        for item in tables:
            if isinstance(item, dict) and isinstance(item.get("table_id"), str):
                table_ids.append(item["table_id"])
    return [table_id for table_id in table_ids if table_id]


if __name__ == "__main__":
    raise SystemExit(main())
