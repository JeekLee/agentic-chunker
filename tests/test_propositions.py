from agentic_chunker._common import Block
from agentic_chunker._propositions import extract


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
