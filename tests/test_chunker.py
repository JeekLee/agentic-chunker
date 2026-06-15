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


def test_new_params_forwarded_to_stages(monkeypatch):
    captured = {}

    def fake_extract(blocks, cfg, **kw):
        from agentic_chunker._common import Proposition
        captured["min_extract_chars"] = kw.get("min_extract_chars")
        captured["extract_concurrency"] = kw.get("concurrency")
        return [Proposition(b.text, b.char_start, b.char_end, b.header) for b in blocks]

    def fake_assign(props, cfg, **kw):
        captured["window_size"] = kw.get("window_size")
        captured["max_props"] = kw.get("max_props")
        captured["assign_concurrency"] = kw.get("concurrency")
        return [Chunk(index=i, text=p.text) for i, p in enumerate(props)]

    monkeypatch.setattr(ac, "_extract", fake_extract)
    monkeypatch.setattr(ac, "_assign", fake_assign)

    chunker = AgenticChunker(
        llm=CFG,
        max_propositions_per_chunk=7,
        window_size=25,
        min_extract_chars=15,
        max_concurrency=3,
    )
    chunker.chunk("# H\n\nAlpha para.")

    assert captured["min_extract_chars"] == 15
    assert captured["extract_concurrency"] == 3
    assert captured["window_size"] == 25
    assert captured["max_props"] == 7
    assert captured["assign_concurrency"] == 3


def test_default_concurrency_and_min_extract_chars(monkeypatch):
    captured = {}

    def fake_extract(blocks, cfg, **kw):
        from agentic_chunker._common import Proposition
        captured["extract_concurrency"] = kw.get("concurrency")
        captured["min_extract_chars"] = kw.get("min_extract_chars")
        return [Proposition(b.text, b.char_start, b.char_end, b.header) for b in blocks]

    def fake_assign(props, cfg, **kw):
        captured["assign_concurrency"] = kw.get("concurrency")
        captured["window_size"] = kw.get("window_size")
        return [Chunk(index=i, text=p.text) for i, p in enumerate(props)]

    monkeypatch.setattr(ac, "_extract", fake_extract)
    monkeypatch.setattr(ac, "_assign", fake_assign)

    AgenticChunker(llm=CFG).chunk("# H\n\nAlpha para.")

    assert captured["extract_concurrency"] == 4      # new default max_concurrency
    assert captured["assign_concurrency"] == 4
    assert captured["min_extract_chars"] == 20       # unchanged default
    assert captured["window_size"] == 40             # unchanged default
