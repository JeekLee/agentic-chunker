from agentic_chunker._common import Proposition
from agentic_chunker._agent import assign


def P(text, header, start=0, end=10):
    return Proposition(text=text, char_start=start, char_end=end, header=header)


def test_single_call_clusters_one_section():
    props = [P("Cats purr.", "Cats", 0, 10), P("Cats sleep.", "Cats", 0, 10),
             P("Dogs bark.", "Cats", 20, 30)]
    calls = []

    def fake_group(texts, cfg):
        calls.append(list(texts))
        return [
            {"proposition_indices": [0, 1], "title": "Cats", "summary": "About cats.", "keywords": ["cats"]},
            {"proposition_indices": [2], "title": "Dogs", "summary": "About dogs.", "keywords": ["dogs"]},
        ]

    chunks = assign(props, cfg=None, group=fake_group)
    assert len(calls) == 1
    assert calls[0] == ["Cats purr.", "Cats sleep.", "Dogs bark."]
    assert [c.text for c in chunks] == ["Cats purr.\nCats sleep.", "Dogs bark."]
    assert [c.index for c in chunks] == [0, 1]
    assert chunks[0].title == "Cats" and chunks[0].keywords == ["cats"]
    assert chunks[0].source_spans == [(0, 10)]
    assert chunks[1].source_spans == [(20, 30)]


def test_sections_grouped_independently_in_order():
    props = [P("Alpha.", "A", 0, 6), P("Beta.", "B", 10, 15)]
    calls = []

    def fake_group(texts, cfg):
        calls.append(list(texts))
        return [{"proposition_indices": [0], "title": "t", "summary": "s", "keywords": []}]

    chunks = assign(props, cfg=None, group=fake_group)
    assert calls == [["Alpha."], ["Beta."]]   # two separate calls, one per section
    assert [c.text for c in chunks] == ["Alpha.", "Beta."]
    assert [c.index for c in chunks] == [0, 1]


def test_large_section_is_split_into_windows():
    props = [P(f"f{i}", "S", 0, 2) for i in range(5)]
    seen = []

    def fake_group(texts, cfg):
        seen.append(list(texts))
        return [{"proposition_indices": list(range(len(texts))),
                 "title": "t", "summary": "s", "keywords": []}]

    chunks = assign(props, cfg=None, group=fake_group, window_size=2, max_props=100)
    assert seen == [["f0", "f1"], ["f2", "f3"], ["f4"]]
    assert [c.text for c in chunks] == ["f0\nf1", "f2\nf3", "f4"]
    assert [c.index for c in chunks] == [0, 1, 2]


def test_max_props_post_cap_splits_oversized_cluster():
    props = [P(f"f{i}", "S", 0, 2) for i in range(5)]

    def fake_group(texts, cfg):
        return [{"proposition_indices": [0, 1, 2, 3, 4], "title": "t", "summary": "s", "keywords": []}]

    chunks = assign(props, cfg=None, group=fake_group, window_size=100, max_props=2)
    assert [c.text for c in chunks] == ["f0\nf1", "f2\nf3", "f4"]
    assert chunks[0].title == "t" and chunks[2].title == "t"


def test_invalid_and_duplicate_indices_dropped():
    props = [P("a", "S", 0, 2), P("b", "S", 4, 6)]

    def fake_group(texts, cfg):
        return [{"proposition_indices": [0, 0, 99], "title": "t", "summary": "s", "keywords": []}]

    chunks = assign(props, cfg=None, group=fake_group)
    assert [c.text for c in chunks] == ["a", "b"]


def test_unassigned_proposition_becomes_own_chunk():
    props = [P("a", "S", 0, 2), P("b", "S", 4, 6)]

    def fake_group(texts, cfg):
        return [{"proposition_indices": [0], "title": "t", "summary": "s", "keywords": []}]

    chunks = assign(props, cfg=None, group=fake_group)
    assert [c.text for c in chunks] == ["a", "b"]
    assert chunks[1].text == "b"


def test_group_failure_falls_back_to_one_chunk_per_proposition():
    props = [P("a", "S", 0, 2), P("b", "S", 4, 6)]

    def fake_group(texts, cfg):
        return None

    chunks = assign(props, cfg=None, group=fake_group)
    assert [c.text for c in chunks] == ["a", "b"]
    assert chunks[0].title == "a" and chunks[0].summary == "a"


def test_empty_props_returns_empty():
    assert assign([], cfg=None, group=lambda t, c: []) == []
