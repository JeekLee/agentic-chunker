import agentic_chunker as ac
from agentic_chunker import AgenticChunker, LlmConfig, Chunk


CFG = LlmConfig(url="http://x/v1", api_key="k", model="m")


def test_chunk_end_to_end_wiring(monkeypatch):
    def fake_group_units(units, cfg, **kw):
        return [Chunk(index=i, text=u.text, title=u.text[:5], summary=u.text,
                      keywords=[], source_spans=u.source_spans)
                for i, u in enumerate(units)]

    monkeypatch.setattr(ac, "_group_units", fake_group_units)

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
    assert hasattr(ac, "DocumentGraph")
    assert hasattr(ac, "DomainSchema")
    assert hasattr(ac, "DomainExtractor")


def test_new_params_forwarded_to_stages(monkeypatch):
    captured = {}

    def fake_group_units(units, cfg, **kw):
        captured["window_size"] = kw.get("window_size")
        captured["max_units"] = kw.get("max_units")
        captured["group_concurrency"] = kw.get("concurrency")
        return [Chunk(index=i, text=u.text) for i, u in enumerate(units)]

    monkeypatch.setattr(ac, "_group_units", fake_group_units)

    chunker = AgenticChunker(
        llm=CFG,
        max_propositions_per_chunk=7,
        window_size=25,
        min_extract_chars=15,
        max_concurrency=3,
    )
    chunker.chunk("# H\n\nAlpha para.")

    assert captured["window_size"] == 25
    assert captured["max_units"] == 7
    assert captured["group_concurrency"] == 3


def test_default_concurrency_and_window_size(monkeypatch):
    captured = {}

    def fake_group_units(units, cfg, **kw):
        captured["group_concurrency"] = kw.get("concurrency")
        captured["window_size"] = kw.get("window_size")
        return [Chunk(index=i, text=u.text) for i, u in enumerate(units)]

    monkeypatch.setattr(ac, "_group_units", fake_group_units)

    AgenticChunker(llm=CFG).chunk("# H\n\nAlpha para.")

    assert captured["group_concurrency"] == 4
    assert captured["window_size"] == 40
