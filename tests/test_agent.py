from agentic_chunker._common import Proposition
from agentic_chunker._agent import assign


def P(text, header, start=0, end=10):
    return Proposition(text=text, char_start=start, char_end=end, header=header)


def test_new_then_existing_builds_one_chunk():
    props = [P("Cats purr.", "Cats", 0, 10), P("Cats sleep a lot.", "Cats", 0, 10)]

    def fake_decide(prop_text, open_chunks, cfg):
        if not open_chunks:
            return {"action": "new", "title": "Cats", "summary": "About cats.", "keywords": ["cats"]}
        return {"action": "existing", "chunk_id": open_chunks[0]["id"],
                "title": "Cats", "summary": "About cats and sleep.", "keywords": ["cats", "sleep"]}

    chunks = assign(props, cfg=None, decide=fake_decide)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.index == 0
    assert c.text == "Cats purr.\nCats sleep a lot."
    assert c.title == "Cats"
    assert c.summary == "About cats and sleep."
    assert c.keywords == ["cats", "sleep"]


def test_action_new_starts_second_chunk():
    props = [P("Cats purr.", "Sec", 0, 10), P("Dogs bark.", "Sec", 0, 10)]

    def fake_decide(prop_text, open_chunks, cfg):
        return {"action": "new", "title": prop_text, "summary": prop_text, "keywords": []}

    chunks = assign(props, cfg=None, decide=fake_decide)
    assert [c.text for c in chunks] == ["Cats purr.", "Dogs bark."]
    assert [c.index for c in chunks] == [0, 1]


def test_sections_never_merge():
    props = [P("Alpha.", "A", 0, 6), P("Beta.", "B", 10, 15)]

    def fake_decide(prop_text, open_chunks, cfg):
        if open_chunks:
            return {"action": "existing", "chunk_id": open_chunks[0]["id"],
                    "title": "x", "summary": "x", "keywords": []}
        return {"action": "new", "title": "x", "summary": "x", "keywords": []}

    chunks = assign(props, cfg=None, decide=fake_decide)
    assert len(chunks) == 2  # one per section, never merged


def test_max_props_forces_new_chunk():
    props = [P(f"Fact {i}.", "Sec", 0, 5) for i in range(3)]

    def fake_decide(prop_text, open_chunks, cfg):
        if open_chunks:
            return {"action": "existing", "chunk_id": open_chunks[0]["id"],
                    "title": "t", "summary": "s", "keywords": []}
        return {"action": "new", "title": "t", "summary": "s", "keywords": []}

    chunks = assign(props, cfg=None, decide=fake_decide, max_props=2)
    assert [len(c.text.split("\n")) for c in chunks] == [2, 1]


def test_source_spans_aggregated_and_deduped():
    props = [P("a", "Sec", 0, 5), P("b", "Sec", 0, 5), P("c", "Sec", 8, 12)]

    def fake_decide(prop_text, open_chunks, cfg):
        if open_chunks:
            return {"action": "existing", "chunk_id": open_chunks[0]["id"],
                    "title": "t", "summary": "s", "keywords": []}
        return {"action": "new", "title": "t", "summary": "s", "keywords": []}

    chunks = assign(props, cfg=None, decide=fake_decide)
    assert chunks[0].source_spans == [(0, 5), (8, 12)]


def test_decide_failure_falls_back_to_new_chunk():
    props = [P("a", "Sec", 0, 5), P("b", "Sec", 0, 5)]

    def fake_decide(prop_text, open_chunks, cfg):
        return None  # simulate LLM/parse failure

    chunks = assign(props, cfg=None, decide=fake_decide)
    assert [c.text for c in chunks] == ["a", "b"]
