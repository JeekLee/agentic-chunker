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
