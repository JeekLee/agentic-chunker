from agentic_chunker._common import Chunk
from agentic_chunker.llm import LlmConfig
import agentic_chunker._unit_agent as unit_agent_mod
from agentic_chunker._unit_agent import group_units


def U(index, text, kind="text", refs=None, linked=None):
    return Chunk(
        index=index,
        text=text,
        title=f"u{index}",
        summary=f"summary {index}",
        keywords=[f"k{index}"],
        source_spans=[(index * 10, index * 10 + len(text))],
        embedding_text=f"hint {index}",
        metadata={
            "common": {"chunk_kind": kind, "section_path": [], "display_format": "plain"},
            "references": {
                "referenced_tables": refs or [],
                "linked_table_indices": linked or [],
                "referenced_by_indices": [],
            },
        },
    )


def test_group_units_preserves_source_text_and_uses_llm_metadata():
    units = [U(0, "원문 A"), U(1, "| 표 |\n| --- |", "table")]

    def fake_group(window, cfg, max_units):
        return [{
            "unit_indices": [0, 1],
            "title": "묶음",
            "summary": "원문 A와 표를 함께 설명한다.",
            "keywords": ["원문", "표"],
            "questions_answered": ["원문 A는 무엇을 설명하나?", "표에는 무엇이 있나?"],
            "embedding_text": "검색용 설명",
        }]

    chunks = group_units(units, cfg=None, group=fake_group)

    assert len(chunks) == 1
    assert chunks[0].text == "원문 A\n\n| 표 |\n| --- |"
    assert chunks[0].title == "묶음"
    assert chunks[0].summary == "원문 A와 표를 함께 설명한다."
    assert chunks[0].keywords == ["원문", "표"]
    assert chunks[0].questions_answered == ["원문 A는 무엇을 설명하나?", "표에는 무엇이 있나?"]
    assert chunks[0].embedding_text == "검색용 설명"
    assert chunks[0].metadata["common"]["chunk_kind"] == "mixed"
    assert chunks[0].metadata["units"] == [
        {"unit_index": 0, "kind": "text", "table_id": ""},
        {"unit_index": 1, "kind": "table", "table_id": ""},
    ]


def test_group_units_converts_linked_table_unit_indices_to_final_chunk_indices():
    parent = U(0, "→ 표 1", refs=["표 1"], linked=[2])
    middle = U(1, "다른 내용")
    table = U(2, "| 표 1 |", "table")
    table.metadata["table"] = {"table_id": "표 1"}
    units = [parent, middle, table]

    def fake_group(window, cfg, max_units):
        return [
            {"unit_indices": [0], "title": "parent", "summary": "", "keywords": []},
            {"unit_indices": [1], "title": "middle", "summary": "", "keywords": []},
            {"unit_indices": [2], "title": "table", "summary": "", "keywords": []},
        ]

    chunks = group_units(units, cfg=None, group=fake_group)

    assert chunks[0].metadata["references"]["referenced_tables"] == ["표 1"]
    assert chunks[0].metadata["references"]["linked_table_indices"] == [2]
    assert chunks[0].metadata["references"]["referenced_table_chunks"] == [
        {"table_id": "표 1", "unit_indices": [2], "chunk_indices": [2]}
    ]
    assert chunks[2].metadata["table"]["table_id"] == "표 1"
    assert chunks[2].metadata["references"]["referenced_by_indices"] == [0]


def test_group_units_preserves_multiple_table_metadata():
    first = U(0, "| A |\n| --- |", "table")
    first.metadata["table"] = {"table_id": "표 1", "headers": ["A"]}
    second = U(1, "| B |\n| --- |", "table")
    second.metadata["table"] = {"table_id": "표 2", "headers": ["B"]}

    def fake_group(window, cfg, max_units):
        return [{"unit_indices": [0, 1], "title": "tables", "summary": "s", "keywords": []}]

    chunks = group_units([first, second], cfg=None, group=fake_group)

    assert "table" not in chunks[0].metadata
    assert chunks[0].metadata["tables"] == [
        {"unit_index": 0, "table_id": "표 1", "headers": ["A"]},
        {"unit_index": 1, "table_id": "표 2", "headers": ["B"]},
    ]


def test_group_units_builds_embedding_text_from_context_when_llm_omits_it():
    units = [U(0, "원문 A")]

    def fake_group(window, cfg, max_units):
        return [{"unit_indices": [0], "title": "제목", "summary": "요약", "keywords": ["키워드"]}]

    chunks = group_units(units, cfg=None, group=fake_group)

    assert chunks[0].embedding_text.startswith("제목: 제목\n요약: 요약\n키워드: 키워드\n")
    assert "hint 0" in chunks[0].embedding_text
    assert chunks[0].questions_answered == ["제목에 대해 무엇을 알 수 있나요?"]


def test_group_units_enriches_fallback_metadata_when_cfg_is_available(monkeypatch):
    units = [U(0, "표 원문", kind="table")]

    def fake_group(window, cfg, max_units):
        return None

    def fake_chat_json(prompt, cfg):
        return {
            "summary": "LLM 요약",
            "keywords": ["LLM", "표"],
            "questions_answered": ["무엇을 설명하나?", "어떤 표인가?"],
        }

    monkeypatch.setattr(unit_agent_mod, "_real_chat_json", fake_chat_json)

    chunks = group_units(
        units,
        cfg=LlmConfig(url="http://x/v1", api_key="k", model="m"),
        group=fake_group,
    )

    assert chunks[0].summary == "LLM 요약"
    assert chunks[0].keywords == ["LLM", "표"]
    assert chunks[0].questions_answered == ["무엇을 설명하나?", "어떤 표인가?"]
    assert chunks[0].metadata["_llm_metadata_generated"] is True


def test_group_units_retries_when_enrichment_returns_too_few_questions(monkeypatch):
    units = [U(0, "짧은 원문", kind="text")]
    replies = [
        {
            "summary": "LLM 요약",
            "keywords": ["짧은 원문"],
            "questions_answered": ["무엇인가?"],
        },
        {
            "questions_answered": ["무엇인가?", "왜 중요한가?", "어떻게 확인하나?"],
        },
    ]

    def fake_group(window, cfg, max_units):
        return None

    def fake_chat_json(prompt, cfg):
        return replies.pop(0)

    monkeypatch.setattr(unit_agent_mod, "_real_chat_json", fake_chat_json)

    chunks = group_units(
        units,
        cfg=LlmConfig(url="http://x/v1", api_key="k", model="m"),
        group=fake_group,
    )

    assert chunks[0].questions_answered == ["무엇인가?", "왜 중요한가?", "어떻게 확인하나?"]
