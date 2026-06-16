from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from agentic_chunker._models import Chunk, DocumentEdge, DocumentGraph, DocumentNode


def _load_evaluator():
    path = Path(__file__).resolve().parents[1] / "examples" / "evaluate_md.py"
    spec = importlib.util.spec_from_file_location("evaluate_md", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lexical_gold_report_uses_table_metadata() -> None:
    evaluator = _load_evaluator()
    chunks = [
        Chunk(index=0, source="다른 본문"),
        Chunk(
            index=1,
            source="| 코드 | 부위 |\n| --- | --- |\n| A | 남성생식기 |",
            title="표 1",
            embedding_text="코드 A 남성생식기",
            metadata={"table": {"table_id": "표 1"}},
        ),
    ]

    report = evaluator._lexical_gold_report(chunks, ["표 1 코드 A::남성생식기"], top_k=1)

    assert report["hit_at_5"] == 1.0
    assert report["queries"][0]["top_chunks"][0]["index"] == 1
    assert report["expanded_hit_at_5"] == 1.0


def test_lexical_gold_report_checks_expanded_graph_context() -> None:
    evaluator = _load_evaluator()
    chunks = [
        Chunk(
            index=0,
            source="자세한 내용은 표 1을 참조한다.",
            document_graph=DocumentGraph(
                nodes=[
                    DocumentNode(id="chunk:0", type="chunk", text="자세한 내용은 표 1을 참조한다."),
                    DocumentNode(id="chunk:1", type="chunk", text="코드 A는 남성생식기 항목이다."),
                ],
                edges=[
                    DocumentEdge("chunk:0", "chunk:1", "REFERS_TO", {"table_id": "표 1"}),
                ],
            ),
        ),
        Chunk(index=1, source="관련 없는 본문"),
    ]

    report = evaluator._lexical_gold_report(chunks, ["표 1 코드 A::남성생식기"], top_k=1)

    assert report["hit_at_5"] == 0.0
    assert report["expanded_hit_at_5"] == 1.0


def test_aggregate_reports_sums_counts_and_averages_ratios() -> None:
    evaluator = _load_evaluator()
    reports = [_report(10, 1.0, 0.5, False), _report(20, 2.0, 1.0, True)]

    aggregate = evaluator._aggregate_reports(reports)

    assert aggregate["files"] == 2
    assert aggregate["input"]["bytes"] == 30
    assert aggregate["speed"]["wall_sec"] == 3.0
    assert aggregate["chunking_quality"]["unit_coverage_ok"] is False
    assert aggregate["chunking_quality"]["files_with_tiny_chunks"] == 2
    assert aggregate["chunking_quality"]["tiny_chunks_by_kind"] == {"text": 2}
    assert aggregate["graph_quality"]["avg_table_reference_coverage"] == 0.75
    assert aggregate["graph_quality"]["files_with_missing_table_refs"] == 1
    assert aggregate["search_quality_expected"]["gold_query_files"] == 2
    assert aggregate["search_quality_expected"]["gold_query_count"] == 2
    assert aggregate["search_quality_expected"]["gold_hit_at_5"] == 0.75
    assert aggregate["search_quality_expected"]["expanded_gold_hit_at_5"] == 0.75


def test_tiny_chunk_details_count_kinds_and_examples() -> None:
    evaluator = _load_evaluator()
    chunks = [
        Chunk(index=0, source="1.", metadata={"common": {"chunk_kind": "text"}}),
        Chunk(index=1, source="| A |\n| --- |", metadata={"common": {"chunk_kind": "table"}}),
        Chunk(index=2, source="충분히 긴 본문입니다. tiny 기준보다 길게 둡니다.", metadata={"common": {"chunk_kind": "text"}}),
    ]

    details = evaluator._tiny_chunk_details(chunks, limit=2)

    assert details["by_kind"] == {"text": 1, "table": 1}
    assert details["examples"] == [
        {"index": 0, "kind": "text", "chars": 2, "title": "", "preview": "1."},
        {"index": 1, "kind": "table", "chars": 13, "title": "", "preview": "| A | | --- |"},
    ]


def test_gold_query_manifest_matches_defaults_paths_and_names(tmp_path: Path) -> None:
    evaluator = _load_evaluator()
    target = tmp_path / "target.md"
    manifest_path = tmp_path / "gold_queries.json"
    manifest_path.write_text(
        json.dumps({
            "default": ["공통 질의::공통"],
            "files": {
                "target.md": ["파일명 질의::파일명"],
                str(target.resolve()): ["절대경로 질의::절대경로"],
            },
        }),
        encoding="utf-8",
    )

    manifest = evaluator._load_gold_query_manifest(manifest_path)
    queries = evaluator._gold_queries_for_path(target, ["CLI 질의::CLI"], manifest)

    assert queries == [
        "CLI 질의::CLI",
        "공통 질의::공통",
        "절대경로 질의::절대경로",
        "파일명 질의::파일명",
    ]


def _report(
    bytes_: int,
    wall_sec: float,
    table_coverage: float,
    unit_coverage_ok: bool,
) -> dict:
    missing_units = [] if unit_coverage_ok else [1]
    return {
        "input": {
            "path": f"/tmp/{bytes_}.md",
            "bytes": bytes_,
            "chars": bytes_,
            "lines": 1,
            "blocks": 1,
            "units": 1,
            "unit_kinds": {"text": 1},
        },
        "speed": {"wall_sec": wall_sec, "llm_calls": 0},
        "chunking_quality": {
            "chunks": 1,
            "tiny_chunks": 1,
            "oversized_chunks": 0,
            "tiny_chunk_details": {"by_kind": {"text": 1}, "examples": []},
            "unit_coverage": {"missing": missing_units, "duplicates": []},
        },
        "graph_quality": {
            "nodes": 1,
            "edges": 1,
            "edge_types": {"HAS_CHUNK": 1},
            "table_reference_coverage": table_coverage,
            "missing_table_reference_edges": missing_units,
        },
        "search_quality_expected": {
            "metadata_complete_ratio": 1.0,
            "chunks_missing_keywords": 0,
            "chunks_with_questions_lt_2": 0,
            "table_context_coverage": table_coverage,
            "lexical_gold_queries": {
                "count": 1,
                "hit_at_5": table_coverage,
                "expanded_hit_at_5": table_coverage,
            },
        },
    }
