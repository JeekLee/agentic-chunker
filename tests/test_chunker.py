import agentic_chunker as ac
from agentic_chunker import AgenticChunker, LlmConfig, Chunk, ChunkingResult


CFG = LlmConfig(url="http://x/v1", api_key="k", model="m")


def test_chunk_end_to_end_wiring(monkeypatch):
    def fake_group_units(units, cfg, **kw):
        return [Chunk(index=i, text=u.text, title=u.text[:5], summary=u.text,
                      keywords=[], source_spans=u.source_spans)
                for i, u in enumerate(units)]

    monkeypatch.setattr("agentic_chunker._chunker._group_units", fake_group_units)

    md = "# H\n\nAlpha para.\n\nBeta para."
    chunker = AgenticChunker(llm=CFG)
    chunks = chunker.chunk(md)

    assert [c.text for c in chunks] == ["Alpha para.", "Beta para."]
    assert all(isinstance(c, Chunk) for c in chunks)
    assert [c.index for c in chunks] == [0, 1]


def test_chunk_document_returns_full_result(monkeypatch):
    def fake_group_units(units, cfg, **kw):
        return [Chunk(index=i, text=u.text, source_spans=u.source_spans) for i, u in enumerate(units)]

    monkeypatch.setattr("agentic_chunker._chunker._group_units", fake_group_units)

    result = AgenticChunker(llm=CFG).chunk_document("# H\n\nAlpha para.\n\nBeta para.")

    assert isinstance(result, ChunkingResult)
    assert [c.source for c in result.chunks] == ["Alpha para.", "Beta para."]
    assert result.document_graph.nodes
    assert result.document_graph.edges
    assert result.entities == []
    assert result.triples == []
    assert result.structured_extractions == []


def test_empty_input_returns_empty_list():
    chunker = AgenticChunker(llm=CFG)
    assert chunker.chunk("") == []
    assert chunker.chunk_document("").chunks == []


def test_exports_available():
    assert hasattr(ac, "AgenticChunker")
    assert hasattr(ac, "LlmConfig")
    assert hasattr(ac, "Chunk")
    assert hasattr(ac, "DocumentGraph")
    assert hasattr(ac, "DomainSchema")
    assert hasattr(ac, "DomainExtractor")
    assert hasattr(ac, "StructuredExtraction")
    assert hasattr(ac, "ChunkingResult")


def test_new_params_forwarded_to_stages(monkeypatch):
    captured = {}

    def fake_group_units(units, cfg, **kw):
        captured["window_size"] = kw.get("window_size")
        captured["max_units"] = kw.get("max_units")
        captured["group_concurrency"] = kw.get("concurrency")
        return [Chunk(index=i, text=u.text) for i, u in enumerate(units)]

    monkeypatch.setattr("agentic_chunker._chunker._group_units", fake_group_units)

    chunker = AgenticChunker(
        llm=CFG,
        max_units_per_chunk=7,
        window_size=25,
        min_extract_chars=15,
        max_concurrency=3,
    )
    chunker.chunk("# H\n\nAlpha para.")

    assert captured["window_size"] == 25
    assert captured["max_units"] == 7
    assert captured["group_concurrency"] == 3


def test_compatibility_max_propositions_param_still_supported(monkeypatch):
    captured = {}

    def fake_group_units(units, cfg, **kw):
        captured["max_units"] = kw.get("max_units")
        return [Chunk(index=i, text=u.text) for i, u in enumerate(units)]

    monkeypatch.setattr("agentic_chunker._chunker._group_units", fake_group_units)

    AgenticChunker(llm=CFG, max_propositions_per_chunk=7).chunk("# H\n\nAlpha para.")

    assert captured["max_units"] == 7


def test_rejects_ambiguous_max_unit_params():
    try:
        AgenticChunker(llm=CFG, max_units_per_chunk=6, max_propositions_per_chunk=7)
    except ValueError as exc:
        assert "cannot both be set" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_default_concurrency_and_window_size(monkeypatch):
    captured = {}

    def fake_group_units(units, cfg, **kw):
        captured["group_concurrency"] = kw.get("concurrency")
        captured["window_size"] = kw.get("window_size")
        return [Chunk(index=i, text=u.text) for i, u in enumerate(units)]

    monkeypatch.setattr("agentic_chunker._chunker._group_units", fake_group_units)

    AgenticChunker(llm=CFG).chunk("# H\n\nAlpha para.")

    assert captured["group_concurrency"] == 4
    assert captured["window_size"] == 40
