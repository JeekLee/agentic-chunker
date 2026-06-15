import agentic_chunker as ac
from agentic_chunker import AgenticChunker, LlmConfig


CFG = LlmConfig(url="http://x/v1", api_key="k", model="m")


def test_agentic_chunker_attaches_document_graph_for_table_reference(monkeypatch):
    def fake_group_units(units, cfg, **kw):
        return list(units)

    monkeypatch.setattr(ac, "_group_units", fake_group_units)

    md = """# Section A

자세한 코드는 → 표 1 참조.

**[표 1]**

| 코드 | 값 |
| --- | --- |
| A | Alpha |
"""
    chunks = AgenticChunker(llm=CFG).chunk(md)

    graph = chunks[0].document_graph
    edge_keys = {(e.source_id, e.target_id, e.type) for e in graph.edges}
    node_types = {n.id: n.type for n in graph.nodes}

    assert chunks[0].source == "자세한 코드는 → 표 1 참조."
    assert ("chunk:0", "chunk:1", "NEXT") in edge_keys
    assert ("chunk:0", "table:unit:1", "REFERS_TO") in edge_keys
    assert ("chunk:1", "table:unit:1", "HAS_TABLE") in edge_keys
    assert node_types["table:unit:1"] == "table"
    assert any(edge.type == "HAS_SECTION" for edge in graph.edges)


def test_document_graph_can_be_disabled(monkeypatch):
    def fake_group_units(units, cfg, **kw):
        return list(units)

    monkeypatch.setattr(ac, "_group_units", fake_group_units)

    chunks = AgenticChunker(llm=CFG, document_graph=False).chunk("Alpha.")

    assert chunks[0].document_graph.nodes == []
    assert chunks[0].document_graph.edges == []
