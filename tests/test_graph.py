import agentic_chunker as ac
from agentic_chunker import AgenticChunker, LlmConfig


CFG = LlmConfig(url="http://x/v1", api_key="k", model="m")


def test_agentic_chunker_attaches_document_graph_for_table_reference(monkeypatch):
    def fake_group_units(units, cfg, **kw):
        return list(units)

    monkeypatch.setattr("agentic_chunker._chunker._group_units", fake_group_units)

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
    nodes = {n.id: n for n in graph.nodes}

    assert chunks[0].source == "자세한 코드는 → 표 1 참조."
    assert chunks[0].source_spans == []
    assert nodes["chunk:0"].metadata["source_spans"]
    assert ("chunk:0", "chunk:1", "NEXT") in edge_keys
    assert ("chunk:0", "table:unit:1", "REFERS_TO") in edge_keys
    assert ("chunk:1", "table:unit:1", "HAS_TABLE") in edge_keys
    assert nodes["table:unit:1"].type == "table"
    assert any(edge.type == "HAS_SECTION" for edge in graph.edges)


def test_graph_can_be_disabled(monkeypatch):
    def fake_group_units(units, cfg, **kw):
        return list(units)

    monkeypatch.setattr("agentic_chunker._chunker._group_units", fake_group_units)

    chunks = AgenticChunker(llm=CFG, document_graph=False).chunk("Alpha.")

    assert chunks[0].document_graph.nodes == []
    assert chunks[0].document_graph.edges == []


def test_table_parts_are_connected_with_continues_edges(monkeypatch):
    def fake_group_units(units, cfg, **kw):
        return list(units)

    monkeypatch.setattr("agentic_chunker._chunker._group_units", fake_group_units)

    rows = "\n".join(f"| {i} | value {i} |" for i in range(13))
    md = f"""**[표 1]**

| 번호 | 값 |
| --- | --- |
{rows}
"""
    chunker = AgenticChunker(llm=CFG)
    chunks = chunker.chunk(md)

    assert len(chunks) == 2
    full_edge_keys = {(edge.source_id, edge.target_id, edge.type) for edge in chunker.document_graph.edges}
    local_edge_keys = {(edge.source_id, edge.target_id, edge.type) for edge in chunks[0].document_graph.edges}
    local_nodes = {node.id for node in chunks[0].document_graph.nodes}

    assert ("table:unit:0", "table:unit:1", "CONTINUES") in full_edge_keys
    assert ("table:unit:0", "table:unit:1", "CONTINUES") in local_edge_keys
    assert "chunk:1" in local_nodes
    assert "table:unit:1" in local_nodes
