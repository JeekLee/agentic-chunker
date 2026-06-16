from agentic_chunker import AgenticChunker, LlmConfig
from agentic_chunker._split import split
from agentic_chunker._tables import split_structured_blocks
from agentic_chunker._units import build_units


CFG = LlmConfig(url="http://x/v1", api_key="k", model="m")


def test_qa_table_is_kept_as_a_markdown_table_chunk():
    md = """| 연번 | 질의 | 답변 |
| --- | --- | --- |
| 1 | Q1 | A1 |
| 2 | Q2 | A2 → 표 1 |
"""
    text_blocks, chunks = split_structured_blocks(split(md))

    assert text_blocks == []
    assert len(chunks) == 1
    assert chunks[0].text == (
        "| 연번 | 질의 | 답변 |\n"
        "| --- | --- | --- |\n"
        "| 1 | Q1 | A1 |\n"
        "| 2 | Q2 | A2 → 표 1 |"
    )
    assert chunks[0].metadata["common"]["chunk_kind"] == "table"
    assert chunks[0].metadata["common"]["display_format"] == "markdown_table"
    assert chunks[0].metadata["table"]["headers"] == ["연번", "질의", "답변"]
    assert "연번=2; 질의=Q2; 답변=A2 → 표 1." in chunks[0].embedding_text
    assert "문답 2. 질문: Q2. 답변: A2 → 표 1." in chunks[0].embedding_text


def test_qa_table_with_spaced_headers_is_kept_as_table():
    md = """| 연번 | 질 의 | 답 변 |
| --- | --- | --- |
| 1 | Q1 | A1 → 표 12 |
"""
    text_blocks, chunks = split_structured_blocks(split(md))

    assert text_blocks == []
    assert len(chunks) == 1
    assert chunks[0].metadata["common"]["chunk_kind"] == "table"
    assert chunks[0].metadata["table"]["headers"] == ["연번", "질 의", "답 변"]
    assert "연번=1; 질 의=Q1; 답 변=A1 → 표 12." in chunks[0].embedding_text
    assert "문답 1. 질문: Q1. 답변: A1 → 표 12." in chunks[0].embedding_text


def test_general_table_keeps_markdown_table_for_display():
    md = """**[표 1]**

| 코드 | 부위 | 코드 | 부위 |
| --- | --- | --- | --- |
| A | 뇌 | H | 남성생식기 |
| B | 안 | I | 여성생식기 |
"""
    text_blocks, chunks = split_structured_blocks(split(md))

    assert text_blocks == []
    assert len(chunks) == 1
    assert chunks[0].text == (
        "**[표 1]**\n\n"
        "| 코드 | 부위 | 코드 | 부위 |\n"
        "| --- | --- | --- | --- |\n"
        "| A | 뇌 | H | 남성생식기 |\n"
        "| B | 안 | I | 여성생식기 |"
    )
    assert chunks[0].title == "표 1"
    assert chunks[0].summary == "표 1 표이며, 2개 행과 코드, 부위, 코드, 부위 열을 포함합니다."
    assert chunks[0].keywords == ["표 1", "코드", "부위"]
    assert chunks[0].metadata["common"]["chunk_kind"] == "table"
    assert chunks[0].metadata["table"]["table_id"] == "표 1"
    assert isinstance(chunks[0].metadata["table"]["source_span"], tuple)
    assert "코드=A; 부위=뇌; 코드=H; 부위=남성생식기." in chunks[0].embedding_text


def test_single_cell_table_artifacts_become_plain_units():
    md = """| 목 차 |
| --- |

| 다음의 수가산정방법 및 청구방법은 세부 내용임. |
| --- |
"""
    text_blocks, chunks = split_structured_blocks(split(md))

    assert text_blocks == []
    assert [chunk.text for chunk in chunks] == [
        "목 차",
        "다음의 수가산정방법 및 청구방법은 세부 내용임.",
    ]
    assert [chunk.metadata["common"]["chunk_kind"] for chunk in chunks] == ["heading", "text"]
    assert [chunk.metadata["common"]["display_format"] for chunk in chunks] == ["plain", "plain"]
    assert "table" not in chunks[0].metadata


def test_empty_table_artifacts_and_placeholders_are_dropped():
    md = """|  |  |
| --- | --- |

⋮

Real text.
"""
    units = build_units(split(md))

    assert [unit.text for unit in units] == ["Real text."]
    assert [unit.metadata["common"]["chunk_kind"] for unit in units] == ["text"]


def test_heading_units_are_prepended_to_following_content():
    md = """| 목 차 |
| --- |

| 연번 | 제목 |
| --- | --- |
| 1 | 일반사항 |

□ 수가산정방법

Body text.
"""
    units = build_units(split(md))

    assert [unit.text for unit in units] == [
        "목 차\n\n| 연번 | 제목 |\n| --- | --- |\n| 1 | 일반사항 |",
        "□ 수가산정방법\n\nBody text.",
    ]
    assert [unit.metadata["common"]["chunk_kind"] for unit in units] == ["table", "text"]
    assert units[0].metadata["common"]["layout_headings"] == ["목 차"]
    assert units[1].metadata["common"]["layout_headings"] == ["□ 수가산정방법"]


def test_agentic_chunker_leaves_table_reference_to_document_graph(monkeypatch):
    captured = {}

    def fake_group_units(units, cfg, **kw):
        captured["kinds"] = [u.metadata["common"]["chunk_kind"] for u in units]
        return list(units)

    monkeypatch.setattr("agentic_chunker._chunker._group_units", fake_group_units)
    md = """| 연번 | 질의 | 답변 |
| --- | --- | --- |
| 7 | 기재방법은? | JS013에 기재함. → 표 1 |

**[표 1]**

| 코드 | 부위 |
| --- | --- |
| A | 뇌 |
"""

    chunks = AgenticChunker(llm=CFG).chunk(md)

    assert captured["kinds"] == ["table", "table"]
    assert [c.metadata["common"]["chunk_kind"] for c in chunks] == ["table", "table"]
    assert "references" not in chunks[0].metadata
    assert chunks[1].text.startswith("**[표 1]**\n\n| 코드 | 부위 |")
    assert any(edge.type == "REFERS_TO" for edge in chunks[0].document_graph.edges)


def test_mixed_text_and_table_sends_units_to_llm(monkeypatch):
    captured = {}

    def fake_group_units(units, cfg, **kw):
        captured["texts"] = [u.text for u in units]
        captured["kinds"] = [u.metadata["common"]["chunk_kind"] for u in units]
        return list(units)

    monkeypatch.setattr("agentic_chunker._chunker._group_units", fake_group_units)

    md = """Intro paragraph.

| 코드 | 부위 |
| --- | --- |
| A | 뇌 |
"""
    chunks = AgenticChunker(llm=CFG).chunk(md)

    assert captured["texts"] == ["Intro paragraph.", "| 코드 | 부위 |\n| --- | --- |\n| A | 뇌 |"]
    assert captured["kinds"] == ["text", "table"]
    assert [c.index for c in chunks] == [0, 1]
    assert chunks[0].text == "Intro paragraph."
    assert chunks[1].metadata["common"]["chunk_kind"] == "table"
