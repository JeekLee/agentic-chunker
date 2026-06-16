"""DocumentGraph generation for final chunks."""
from __future__ import annotations

import re

from ._models import Chunk, DocumentEdge, DocumentGraph, DocumentNode
from ._references import table_references


def attach_document_graphs(chunks: list[Chunk], *, document_id: str = "document:0") -> DocumentGraph:
    """Build a document graph and attach each chunk's local neighborhood graph."""
    full_graph = build_document_graph(chunks, document_id=document_id)
    for chunk in chunks:
        chunk.document_graph = local_document_graph(full_graph, chunk.id)
        chunk.source_spans = []
    return full_graph


def build_document_graph(chunks: list[Chunk], *, document_id: str = "document:0") -> DocumentGraph:
    nodes: dict[str, DocumentNode] = {}
    edges: list[DocumentEdge] = []
    edge_keys: set[tuple[str, str, str]] = set()
    section_ids: dict[str, str] = {}
    table_id_to_nodes: dict[str, list[tuple[str, dict]]] = {}
    table_sequence_nodes: dict[tuple, list[tuple[int, str, dict]]] = {}

    def add_node(node: DocumentNode) -> None:
        nodes.setdefault(node.id, node)

    def add_edge(source_id: str, target_id: str, edge_type: str, metadata: dict | None = None) -> None:
        key = (source_id, target_id, edge_type)
        if key in edge_keys:
            return
        edge_keys.add(key)
        edges.append(DocumentEdge(source_id, target_id, edge_type, dict(metadata or {})))

    add_node(DocumentNode(id=document_id, type="document"))

    for chunk in chunks:
        chunk.id = f"chunk:{chunk.index}"
        add_node(DocumentNode(
            id=chunk.id,
            type="chunk",
            text=chunk.source,
            metadata={
                "index": chunk.index,
                "title": chunk.title,
                "source_spans": chunk.source_spans,
                "common": chunk.metadata.get("common", {}),
            },
        ))

        section_id = _chunk_section_id(chunk, section_ids)
        if section_id:
            section_name = chunk.metadata.get("common", {}).get("section_path", [""])[-1]
            add_node(DocumentNode(id=section_id, type="section", text=section_name))
            add_edge(document_id, section_id, "HAS_SECTION")
            add_edge(section_id, chunk.id, "HAS_CHUNK")
        else:
            add_edge(document_id, chunk.id, "HAS_CHUNK")

        for table_no, table in enumerate(_chunk_tables(chunk)):
            node_id = _table_node_id(chunk, table, table_no)
            add_node(DocumentNode(
                id=node_id,
                type="table",
                text=None,
                metadata={**table, "source_chunk_id": chunk.id},
            ))
            add_edge(chunk.id, node_id, "HAS_TABLE", {"table_id": table.get("table_id", "")})
            table_id = table.get("table_id")
            if isinstance(table_id, str) and table_id:
                table_id_to_nodes.setdefault(table_id, []).append((node_id, table))
            sequence_key = _table_sequence_key(table)
            part_index = table.get("part_index")
            if sequence_key is not None and isinstance(part_index, int):
                table_sequence_nodes.setdefault(sequence_key, []).append((part_index, node_id, table))

    for prev, curr in zip(chunks, chunks[1:]):
        add_edge(prev.id, curr.id, "NEXT")
        add_edge(curr.id, prev.id, "PREVIOUS")

    for sequence in table_sequence_nodes.values():
        ordered = sorted(sequence, key=lambda item: item[0])
        for (_, source_id, source_table), (target_part, target_id, target_table) in zip(ordered, ordered[1:]):
            add_edge(source_id, target_id, "CONTINUES", {
                "table_id": source_table.get("table_id", ""),
                "part_index": target_part,
                "part_total": target_table.get("part_total"),
            })

    for chunk in chunks:
        for table_id in table_references(chunk.source):
            for node_id, table in table_id_to_nodes.get(table_id, []):
                add_edge(chunk.id, node_id, "REFERS_TO", {
                    "table_id": table_id,
                    "unit_index": table.get("unit_index"),
                })

    return DocumentGraph(nodes=list(nodes.values()), edges=edges)


def local_document_graph(graph: DocumentGraph, center_chunk_id: str) -> DocumentGraph:
    node_by_id = {node.id: node for node in graph.nodes}
    included_ids = {center_chunk_id}
    included_edges: list[DocumentEdge] = []

    for edge in graph.edges:
        if edge.source_id == center_chunk_id or edge.target_id == center_chunk_id:
            included_edges.append(edge)
            included_ids.add(edge.source_id)
            included_ids.add(edge.target_id)

    # Keep document/section path for the current chunk, and source chunk for
    # referenced table nodes so a retriever can expand to the owning chunk.
    changed = True
    while changed:
        changed = False
        for edge in graph.edges:
            if edge.target_id in included_ids and edge.type in {"HAS_SECTION", "HAS_CHUNK", "HAS_TABLE"}:
                if edge not in included_edges:
                    included_edges.append(edge)
                if edge.source_id not in included_ids:
                    included_ids.add(edge.source_id)
                    changed = True

    for node_id in list(included_ids):
        node = node_by_id.get(node_id)
        source_chunk_id = node.metadata.get("source_chunk_id") if node else None
        if isinstance(source_chunk_id, str) and source_chunk_id not in included_ids:
            included_ids.add(source_chunk_id)
            for edge in graph.edges:
                if edge.source_id == source_chunk_id and edge.target_id == node_id:
                    included_edges.append(edge)

    continuation_changed = True
    while continuation_changed:
        continuation_changed = False
        for edge in graph.edges:
            if edge.type != "CONTINUES":
                continue
            if edge.source_id in included_ids or edge.target_id in included_ids:
                if edge not in included_edges:
                    included_edges.append(edge)
                for node_id in (edge.source_id, edge.target_id):
                    if node_id not in included_ids:
                        included_ids.add(node_id)
                        continuation_changed = True

    for node_id in list(included_ids):
        node = node_by_id.get(node_id)
        source_chunk_id = node.metadata.get("source_chunk_id") if node else None
        if isinstance(source_chunk_id, str) and source_chunk_id not in included_ids:
            included_ids.add(source_chunk_id)
            for edge in graph.edges:
                if edge.source_id == source_chunk_id and edge.target_id == node_id:
                    included_edges.append(edge)

    nodes = [node for node in graph.nodes if node.id in included_ids]
    return DocumentGraph(nodes=nodes, edges=_dedupe_edges(included_edges))


def _chunk_section_id(chunk: Chunk, section_ids: dict[str, str]) -> str:
    section_path = chunk.metadata.get("common", {}).get("section_path", [])
    if not section_path:
        return ""
    section_name = str(section_path[-1])
    if section_name not in section_ids:
        section_ids[section_name] = f"section:{len(section_ids)}:{_slug(section_name)}"
    return section_ids[section_name]


def _chunk_tables(chunk: Chunk) -> list[dict]:
    if "tables" in chunk.metadata and isinstance(chunk.metadata["tables"], list):
        return [dict(table) for table in chunk.metadata["tables"] if isinstance(table, dict)]
    if "table" in chunk.metadata and isinstance(chunk.metadata["table"], dict):
        table = dict(chunk.metadata["table"])
        table.setdefault("unit_index", _table_unit_index(chunk, table))
        return [table]
    return []


def _table_unit_index(chunk: Chunk, table: dict) -> int | None:
    table_id = table.get("table_id", "")
    for unit in chunk.metadata.get("units", []):
        if not isinstance(unit, dict):
            continue
        if unit.get("kind") not in {"table", "table_part", "table_caption"}:
            continue
        if unit.get("table_id", "") == table_id and isinstance(unit.get("unit_index"), int):
            return unit["unit_index"]
    return chunk.index if chunk.metadata.get("common", {}).get("chunk_kind") in {"table", "table_part", "table_caption"} else None


def _table_node_id(chunk: Chunk, table: dict, table_no: int) -> str:
    unit_index = table.get("unit_index")
    if isinstance(unit_index, int):
        return f"table:unit:{unit_index}"
    table_id = table.get("table_id")
    if table_id:
        return f"table:{_slug(str(table_id))}:{chunk.index}:{table_no}"
    return f"table:{chunk.index}:{table_no}"


def _table_sequence_key(table: dict) -> tuple | None:
    part_total = table.get("part_total")
    if not isinstance(part_total, int) or part_total <= 1:
        return None
    table_id = table.get("table_id")
    if isinstance(table_id, str) and table_id:
        return ("table_id", table_id)
    source_span = table.get("source_span")
    if isinstance(source_span, (list, tuple)) and len(source_span) == 2:
        return ("source_span", tuple(source_span))
    return None


def _slug(text: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z가-힣]+", "-", text).strip("-")
    return slug or "node"


def _dedupe_edges(edges: list[DocumentEdge]) -> list[DocumentEdge]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[DocumentEdge] = []
    for edge in edges:
        key = (edge.source_id, edge.target_id, edge.type)
        if key in seen:
            continue
        seen.add(key)
        unique.append(edge)
    return unique
