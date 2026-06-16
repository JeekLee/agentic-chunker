from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from agentic_chunker._models import Chunk


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


def test_aggregate_reports_sums_counts_and_averages_ratios() -> None:
    evaluator = _load_evaluator()
    reports = [_report(10, 1.0, 0.5, False), _report(20, 2.0, 1.0, True)]

    aggregate = evaluator._aggregate_reports(reports)

    assert aggregate["files"] == 2
    assert aggregate["input"]["bytes"] == 30
    assert aggregate["speed"]["wall_sec"] == 3.0
    assert aggregate["chunking_quality"]["unit_coverage_ok"] is False
    assert aggregate["graph_quality"]["avg_table_reference_coverage"] == 0.75
    assert aggregate["graph_quality"]["files_with_missing_table_refs"] == 1
    assert aggregate["search_quality_expected"]["gold_query_files"] == 2
    assert aggregate["search_quality_expected"]["gold_query_count"] == 2
    assert aggregate["search_quality_expected"]["gold_hit_at_5"] == 0.75


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
            "tiny_chunks": 0,
            "oversized_chunks": 0,
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
            "lexical_gold_queries": {"count": 1, "hit_at_5": table_coverage},
        },
    }
