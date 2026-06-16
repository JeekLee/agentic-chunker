from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

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
    reports = [
        _report(10, 1.0, 0.5, False, max_units=8, window_size=10, max_concurrency=1),
        _report(20, 2.0, 1.0, True, llm_calls=2, max_units=12, window_size=20, max_concurrency=4),
    ]

    aggregate = evaluator._aggregate_reports(reports)

    assert aggregate["files"] == 2
    assert aggregate["config"] == {
        "profiles": ["custom"],
        "modes": ["deterministic", "llm"],
        "models": ["m"],
        "max_units": [8, 12],
        "window_size": [10, 20],
        "max_concurrency": [1, 4],
        "max_good_source_chars": [6000],
    }
    assert aggregate["input"]["bytes"] == 30
    assert aggregate["speed"]["wall_sec"] == 3.0
    assert aggregate["speed"]["llm_calls"] == 2
    assert aggregate["speed"]["llm_failed"] == 1
    assert aggregate["speed"]["llm_call_summary"]["group"] == {
        "count": 2,
        "ok": 1,
        "failed": 1,
        "avg_sec": 2.0,
        "max_sec": 3.0,
        "avg_prompt_chars": 150,
        "max_prompt_chars": 200,
    }
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


def test_named_profile_full_no_llm_sets_benchmark_defaults() -> None:
    evaluator = _load_evaluator()
    args = _args(profile="full-no-llm", paths=[Path("a.md")])

    evaluator._prepare_args(args)

    assert args.no_llm is True
    assert args.aggregate_only is True
    assert args.benchmark_profile == "full-no-llm"


def test_named_profile_llm_smoke_sets_speed_defaults() -> None:
    evaluator = _load_evaluator()
    args = _args(profile="llm-smoke", paths=[Path("a.md")])

    evaluator._prepare_args(args)

    assert args.no_llm is False
    assert args.aggregate_only is True
    assert args.window_size == 20
    assert args.benchmark_profile == "llm-smoke"


def test_profile_file_window_size_overrides_llm_smoke_default(tmp_path: Path) -> None:
    evaluator = _load_evaluator()
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps({
            "profile": "llm-smoke",
            "paths": ["a.md"],
            "window_size": 40,
        }),
        encoding="utf-8",
    )
    args = _args(profile_file=profile_path)

    evaluator._prepare_args(args)

    assert args.profile == "llm-smoke"
    assert args.window_size == 40


def test_cli_window_size_overrides_llm_smoke_default() -> None:
    evaluator = _load_evaluator()
    args = _args(profile="llm-smoke", paths=[Path("a.md")], window_size=40)

    evaluator._prepare_args(args)

    assert args.window_size == 40


def test_profile_file_overrides_paths_mode_and_gold_queries(tmp_path: Path) -> None:
    evaluator = _load_evaluator()
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    profile_path = tmp_path / "profile.json"
    gold_path = tmp_path / "gold.json"
    save_path = tmp_path / "current.json"
    compare_path = tmp_path / "baseline.json"
    profile_path.write_text(
        json.dumps({
            "name": "mdout-smoke",
            "mode": "llm",
            "paths": [str(first), str(second)],
            "gold_queries": ["질의::정답"],
            "gold_query_file": str(gold_path),
            "save_report": str(save_path),
            "compare_report": str(compare_path),
            "aggregate_only": True,
            "max_concurrency": 2,
            "timeout": 90,
        }),
        encoding="utf-8",
    )
    args = _args(profile_file=profile_path)

    evaluator._prepare_args(args)

    assert args.paths == [first, second]
    assert args.no_llm is False
    assert args.gold_query == ["질의::정답"]
    assert args.gold_query_file == gold_path
    assert args.save_report == save_path
    assert args.compare_report == compare_path
    assert args.aggregate_only is True
    assert args.max_concurrency == 2
    assert args.timeout == 90
    assert args.benchmark_profile == "mdout-smoke"


def test_profile_file_rejects_boolean_integer_options(tmp_path: Path) -> None:
    evaluator = _load_evaluator()
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps({
            "paths": ["a.md"],
            "timeout": True,
        }),
        encoding="utf-8",
    )
    args = _args(profile_file=profile_path)

    try:
        evaluator._prepare_args(args)
    except SystemExit as exc:
        assert "timeout" in str(exc)
    else:
        raise AssertionError("boolean timeout should be rejected")


def test_compare_reports_groups_metric_statuses_and_candidates() -> None:
    evaluator = _load_evaluator()
    baseline = evaluator._aggregate_reports([
        _report(10, 10.0, 0.5, True),
        _report(20, 10.0, 1.0, True),
    ])
    current = json.loads(json.dumps(baseline))
    current["speed"]["wall_sec"] = 12.0
    current["chunking_quality"]["tiny_chunks"] = 1
    current["graph_quality"]["avg_table_reference_coverage"] = 1.0
    current["search_quality_expected"]["chunks_missing_keywords"] = 2

    comparison = evaluator._compare_reports(baseline, current)

    assert comparison["areas"]["speed"]["wall_sec"]["status"] == "improved"
    assert comparison["areas"]["chunking_quality"]["tiny_chunks"]["status"] == "improved"
    assert comparison["areas"]["graph_quality"]["avg_table_reference_coverage"]["status"] == "improved"
    assert comparison["areas"]["search_quality_expected"]["chunks_missing_keywords"]["status"] == "regressed"
    assert comparison["summary"]["regressed"] == 1
    assert comparison["improvement_candidates"][0] == {
        "area": "search_quality_expected",
        "metric": "chunks_missing_keywords",
        "current": 2,
        "baseline": 0,
        "reason": "lower is better but current value increased",
    }


def test_compare_reports_ignores_small_speed_jitter() -> None:
    evaluator = _load_evaluator()
    baseline = evaluator._aggregate_reports([_report(10, 0.333, 1.0, True)])
    current = json.loads(json.dumps(baseline))
    current["speed"]["wall_sec"] = 0.342

    comparison = evaluator._compare_reports(baseline, current)

    assert comparison["areas"]["speed"]["wall_sec"]["status"] == "unchanged"
    assert comparison["summary"]["regressed"] == 0
    assert comparison["improvement_candidates"] == []


def test_benchmark_subject_prefers_aggregate_payload() -> None:
    evaluator = _load_evaluator()
    aggregate = {"files": 2, "speed": {"wall_sec": 1.0}}
    payload = {"aggregate": aggregate, "files": [{"speed": {"wall_sec": 2.0}}]}

    assert evaluator._benchmark_subject(payload) is aggregate


def _args(**overrides):
    values = {
        "paths": [],
        "encoding": "utf-8",
        "profile": "custom",
        "profile_file": None,
        "no_llm": False,
        "aggregate_only": False,
        "llm_url": "",
        "llm_api_key": "",
        "llm_model": "",
        "timeout": 180,
        "max_units": 8,
        "window_size": None,
        "max_concurrency": 4,
        "max_good_source_chars": 6000,
        "gold_query": [],
        "gold_query_file": None,
        "save_report": None,
        "compare_report": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _report(
    bytes_: int,
    wall_sec: float,
    table_coverage: float,
    unit_coverage_ok: bool,
    llm_calls: int = 0,
    max_units: int = 8,
    window_size: int = 10,
    max_concurrency: int = 4,
    max_good_source_chars: int = 6000,
) -> dict:
    missing_units = [] if unit_coverage_ok else [1]
    llm_call_summary = {}
    if llm_calls:
        llm_call_summary = {
            "group": {
                "count": 2,
                "ok": 1,
                "failed": 1,
                "avg_sec": 2.0,
                "max_sec": 3.0,
                "avg_prompt_chars": 150,
                "max_prompt_chars": 200,
            },
        }
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
        "config": {
            "profile": "custom",
            "mode": "llm" if llm_calls else "deterministic",
            "model": "m" if llm_calls else None,
            "max_units": max_units,
            "window_size": window_size,
            "max_concurrency": max_concurrency,
            "max_good_source_chars": max_good_source_chars,
        },
        "speed": {"wall_sec": wall_sec, "llm_calls": llm_calls, "llm_call_summary": llm_call_summary},
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
