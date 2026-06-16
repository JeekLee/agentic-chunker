from dataclasses import asdict

from agentic_chunker._models import Block, Proposition, Chunk, DocumentGraph


def test_block_holds_text_offsets_and_header():
    b = Block(text="hello", char_start=0, char_end=5, header="Intro")
    assert b.text == "hello"
    assert (b.char_start, b.char_end) == (0, 5)
    assert b.header == "Intro"


def test_proposition_carries_source_span_and_header():
    p = Proposition(text="X is Y.", char_start=10, char_end=20, header="Intro")
    assert p.text == "X is Y."
    assert (p.char_start, p.char_end) == (10, 20)
    assert p.header == "Intro"
    assert p.source_text == ""


def test_chunk_has_all_output_fields_with_defaults():
    c = Chunk(index=0, text="X is Y.")
    assert c.index == 0
    assert c.source == "X is Y."
    assert c.text == "X is Y."
    assert c.title == ""
    assert c.summary == ""
    assert c.keywords == []
    assert c.questions_answered == []
    assert isinstance(c.document_graph, DocumentGraph)
    assert c.source_spans == []
    assert c.embedding_text == ""
    assert c.metadata == {}


def test_chunk_dataclass_serializes_public_payload_only():
    c = Chunk(
        index=7,
        text="X is Y.",
        summary="s",
        keywords=["x"],
        questions_answered=["What is X?"],
        title="internal",
        metadata={"internal": True},
    )

    assert asdict(c).keys() == {
        "source",
        "summary",
        "keywords",
        "questions_answered",
        "document_graph",
    }
