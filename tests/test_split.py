from agentic_chunker._split import split


def test_empty_input_yields_no_blocks():
    assert split("") == []
    assert split("   \n\n  ") == []


def test_single_paragraph_no_header():
    blocks = split("Just one paragraph of text.")
    assert len(blocks) == 1
    b = blocks[0]
    assert b.text == "Just one paragraph of text."
    assert b.header is None
    assert b.char_start == 0
    assert b.char_end == len("Just one paragraph of text.")


def test_paragraphs_split_on_blank_lines():
    blocks = split("First para.\n\nSecond para.")
    assert [b.text for b in blocks] == ["First para.", "Second para."]


def test_header_assigned_to_following_blocks():
    md = "# Intro\n\nAlpha para.\n\n## Details\n\nBeta para."
    blocks = split(md)
    texts = [(b.header, b.text) for b in blocks]
    assert texts == [("Intro", "Alpha para."), ("Details", "Beta para.")]


def test_header_line_is_not_emitted_as_a_block():
    blocks = split("# Only A Header\n")
    assert blocks == []


def test_char_offsets_point_into_source():
    md = "# H\n\nHello there."
    blocks = split(md)
    assert len(blocks) == 1
    b = blocks[0]
    assert md[b.char_start:b.char_end] == "Hello there."
