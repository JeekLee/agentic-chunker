from agentic_chunker._models import Block
from agentic_chunker._legacy_propositions import extract


def make_blocks():
    return [
        Block(text="Cats purr. They also sleep.", char_start=0, char_end=27, header="Cats"),
        Block(text="Dogs bark.", char_start=30, char_end=40, header="Dogs"),
    ]


def test_extract_returns_one_proposition_per_returned_item():
    calls = []

    def fake_chat_json(prompt, cfg):
        calls.append(prompt)
        if "Cats purr" in prompt:
            return ["Cats purr.", "Cats sleep."]
        return ["Dogs bark."]

    props = extract(make_blocks(), cfg=None, chat_json=fake_chat_json)
    assert [p.text for p in props] == ["Cats purr.", "Cats sleep.", "Dogs bark."]


def test_extracted_propositions_inherit_block_span_and_header():
    def fake_chat_json(prompt, cfg):
        if "Cats purr" in prompt:
            return ["Cats purr.", "Cats sleep."]
        return ["Dogs bark."]

    props = extract(make_blocks(), cfg=None, chat_json=fake_chat_json)
    cats = [p for p in props if p.header == "Cats"]
    assert all(p.char_start == 0 and p.char_end == 27 for p in cats)
    dogs = [p for p in props if p.header == "Dogs"]
    assert dogs[0].char_start == 30 and dogs[0].char_end == 40


def test_fallback_uses_block_text_when_llm_returns_none():
    def fake_chat_json(prompt, cfg):
        return None

    props = extract(make_blocks(), cfg=None, chat_json=fake_chat_json)
    assert [p.text for p in props] == ["Cats purr. They also sleep.", "Dogs bark."]


def test_non_string_items_are_ignored():
    def fake_chat_json(prompt, cfg):
        return ["good", 123, {"x": 1}, "  "]

    props = extract([make_blocks()[0]], cfg=None, chat_json=fake_chat_json)
    assert [p.text for p in props] == ["good"]


def test_all_junk_items_fall_back_to_block_text():
    def fake_chat_json(prompt, cfg):
        return ["  ", 123, {"x": 1}]

    props = extract([make_blocks()[0]], cfg=None, chat_json=fake_chat_json)
    assert [p.text for p in props] == ["Cats purr. They also sleep."]


def test_block_text_with_braces_does_not_crash():
    block = Block(text='config = {"a": 1} and {b}', char_start=0, char_end=25, header=None)

    def fake_chat_json(prompt, cfg):
        # The full block text (including braces) must be passed through intact.
        assert 'config = {"a": 1} and {b}' in prompt
        return ["config has a=1"]

    props = extract([block], cfg=None, chat_json=fake_chat_json)
    assert [p.text for p in props] == ["config has a=1"]


def test_chat_json_exception_falls_back_to_block_text():
    def fake_chat_json(prompt, cfg):
        raise RuntimeError("boom")

    props = extract([make_blocks()[1]], cfg=None, chat_json=fake_chat_json)
    assert [p.text for p in props] == ["Dogs bark."]


def test_short_block_skips_llm_and_is_verbatim():
    called = {"n": 0}

    def fake_chat_json(prompt, cfg):
        called["n"] += 1
        return ["should not be used"]

    short = Block(text="목적", char_start=0, char_end=2, header="H")
    props = extract([short], cfg=None, chat_json=fake_chat_json, min_extract_chars=20)
    assert called["n"] == 0                      # no LLM call for a short block
    assert [p.text for p in props] == ["목적"]   # emitted verbatim
    assert props[0].char_start == 0 and props[0].char_end == 2 and props[0].header == "H"


def test_block_at_threshold_length_is_extracted():
    # len("x" * 20) == 20, which is NOT < 20, so it must be extracted.
    called = {"n": 0}

    def fake_chat_json(prompt, cfg):
        called["n"] += 1
        return ["p1", "p2"]

    blk = Block(text="x" * 20, char_start=0, char_end=20, header=None)
    props = extract([blk], cfg=None, chat_json=fake_chat_json, min_extract_chars=20)
    assert called["n"] == 1
    assert [p.text for p in props] == ["p1", "p2"]


def test_min_extract_chars_defaults_to_20():
    called = {"n": 0}

    def fake_chat_json(prompt, cfg):
        called["n"] += 1
        return ["x"]

    # 19-char block, default threshold -> skipped
    blk = Block(text="a" * 19, char_start=0, char_end=19, header=None)
    props = extract([blk], cfg=None, chat_json=fake_chat_json)
    assert called["n"] == 0
    assert [p.text for p in props] == ["a" * 19]
