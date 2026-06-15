from agentic_chunker import AgenticChunker, LlmConfig
from agentic_chunker._common import Chunk
from agentic_chunker._split import split
from agentic_chunker._tables import link_table_references
from agentic_chunker._tables import split_structured_blocks


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
    assert "코드=A; 부위=뇌; 코드=H; 부위=남성생식기." in chunks[0].embedding_text


def test_agentic_chunker_links_qa_reference_to_table_chunk(monkeypatch):
    captured = {}

    def fake_group_units(units, cfg, **kw):
        captured["kinds"] = [u.metadata["common"]["chunk_kind"] for u in units]
        return list(units)

    monkeypatch.setattr("agentic_chunker._group_units", fake_group_units)
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
    assert chunks[0].metadata["references"]["referenced_tables"] == ["표 1"]
    assert chunks[0].metadata["references"]["linked_table_indices"] == [1]
    assert chunks[1].text.startswith("**[표 1]**\n\n| 코드 | 부위 |")


def test_mixed_text_and_table_only_sends_text_to_llm(monkeypatch):
    captured = {}

    def fake_group_units(units, cfg, **kw):
        captured["texts"] = [u.text for u in units]
        captured["kinds"] = [u.metadata["common"]["chunk_kind"] for u in units]
        return list(units)

    monkeypatch.setattr("agentic_chunker._group_units", fake_group_units)

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


def test_references_are_linked_for_text_and_general_table_chunks():
    chunks = [
        Chunk(index=0, text="자세한 코드는 → 표 12 참조.", metadata={"common": {"chunk_kind": "text"}}),
        Chunk(
            index=1,
            text="| 구분 | 설명 |\n| --- | --- |\n| 급여 | 표 12를 따른다 |",
            metadata={"common": {"chunk_kind": "table"}, "table": {"table_id": ""}},
        ),
        Chunk(
            index=2,
            text="| 코드 | 값 |\n| --- | --- |\n| EB443 | 충수 |",
            metadata={"common": {"chunk_kind": "table_part"}, "table": {"table_id": "표 12"}},
        ),
        Chunk(
            index=3,
            text="| 코드 | 값 |\n| --- | --- |\n| EB444 | 소장·대장 |",
            metadata={"common": {"chunk_kind": "table_part"}, "table": {"table_id": "표 12"}},
        ),
    ]

    link_table_references(chunks)

    assert chunks[0].metadata["references"]["referenced_tables"] == ["표 12"]
    assert chunks[0].metadata["references"]["linked_table_indices"] == [2, 3]
    assert chunks[1].metadata["references"]["referenced_tables"] == ["표 12"]
    assert chunks[1].metadata["references"]["linked_table_indices"] == [2, 3]
    assert chunks[2].metadata["references"]["referenced_tables"] == []
    assert chunks[2].metadata["references"]["linked_table_indices"] == []


def test_table_caption_does_not_create_self_reference():
    chunks = [
        Chunk(
            index=0,
            text="**[표 12]**\n\n| 코드 | 값 |\n| --- | --- |\n| EB443 | 충수 |",
            metadata={"common": {"chunk_kind": "table"}, "table": {"table_id": "표 12"}},
        ),
    ]

    link_table_references(chunks)

    assert chunks[0].metadata["references"]["referenced_tables"] == []
    assert chunks[0].metadata["references"]["linked_table_indices"] == []
