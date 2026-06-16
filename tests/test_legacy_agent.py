from agentic_chunker._models import Proposition
from agentic_chunker._legacy_agent import assign


def P(text, header, start=0, end=10):
    return Proposition(text=text, char_start=start, char_end=end, header=header)


def test_single_call_clusters_one_section():
    props = [P("Cats purr.", "Cats", 0, 10), P("Cats sleep.", "Cats", 0, 10),
             P("Dogs bark.", "Cats", 20, 30)]
    calls = []

    def fake_group(texts, cfg, max_props):
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

    def fake_group(texts, cfg, max_props):
        calls.append(list(texts))
        return [{"proposition_indices": [0], "title": "t", "summary": "s", "keywords": []}]

    chunks = assign(props, cfg=None, group=fake_group)
    assert calls == [["Alpha."], ["Beta."]]
    assert [c.text for c in chunks] == ["Alpha.", "Beta."]
    assert [c.index for c in chunks] == [0, 1]


def test_large_section_is_split_into_windows():
    props = [P(f"f{i}", "S", 0, 2) for i in range(5)]
    seen = []

    def fake_group(texts, cfg, max_props):
        seen.append(list(texts))
        return [{"proposition_indices": list(range(len(texts))),
                 "title": "t", "summary": "s", "keywords": []}]

    chunks = assign(props, cfg=None, group=fake_group, window_size=2, max_props=100)
    assert seen == [["f0", "f1"], ["f2", "f3"], ["f4"]]
    assert [c.text for c in chunks] == ["f0\nf1", "f2\nf3", "f4"]
    assert [c.index for c in chunks] == [0, 1, 2]
    assert [c.title for c in chunks] == ["t", "t", "t"]


def test_max_props_post_cap_splits_with_part_markers():
    props = [P(f"f{i}", "S", 0, 2) for i in range(5)]

    def fake_group(texts, cfg, max_props):
        return [{"proposition_indices": [0, 1, 2, 3, 4], "title": "t", "summary": "s", "keywords": []}]

    chunks = assign(props, cfg=None, group=fake_group, window_size=100, max_props=2)
    assert [c.text for c in chunks] == ["f0\nf1", "f2\nf3", "f4"]
    assert [c.title for c in chunks] == ["t (1/3)", "t (2/3)", "t (3/3)"]
    assert all(c.summary == "s" for c in chunks)


def test_single_part_cluster_has_no_marker():
    props = [P("a", "S", 0, 2), P("b", "S", 4, 6)]

    def fake_group(texts, cfg, max_props):
        return [{"proposition_indices": [0, 1], "title": "t", "summary": "s", "keywords": []}]

    chunks = assign(props, cfg=None, group=fake_group, max_props=10)
    assert len(chunks) == 1
    assert chunks[0].title == "t"


def test_max_props_is_passed_to_group():
    props = [P("a", "S", 0, 2)]
    seen = {}

    def fake_group(texts, cfg, max_props):
        seen["max_props"] = max_props
        return [{"proposition_indices": [0], "title": "t", "summary": "s", "keywords": []}]

    assign(props, cfg=None, group=fake_group, max_props=7)
    assert seen["max_props"] == 7


def test_invalid_and_duplicate_indices_dropped():
    props = [P("a", "S", 0, 2), P("b", "S", 4, 6)]

    def fake_group(texts, cfg, max_props):
        return [{"proposition_indices": [0, 0, 99], "title": "t", "summary": "s", "keywords": []}]

    chunks = assign(props, cfg=None, group=fake_group)
    assert [c.text for c in chunks] == ["a", "b"]


def test_unassigned_proposition_becomes_own_chunk():
    props = [P("a", "S", 0, 2), P("b", "S", 4, 6)]

    def fake_group(texts, cfg, max_props):
        return [{"proposition_indices": [0], "title": "t", "summary": "s", "keywords": []}]

    chunks = assign(props, cfg=None, group=fake_group)
    assert [c.text for c in chunks] == ["a", "b"]
    assert chunks[1].text == "b"


def test_group_failure_falls_back_to_one_chunk_per_proposition():
    props = [P("a", "S", 0, 2), P("b", "S", 4, 6)]

    def fake_group(texts, cfg, max_props):
        return None

    chunks = assign(props, cfg=None, group=fake_group)
    assert [c.text for c in chunks] == ["a", "b"]
    assert chunks[0].title == "a" and chunks[0].summary == "a"


def test_empty_props_returns_empty():
    assert assign([], cfg=None, group=lambda t, c, m: []) == []


def test_group_empty_list_falls_back_to_own_chunks():
    props = [P("a", "S", 0, 2), P("b", "S", 4, 6)]

    def fake_group(texts, cfg, max_props):
        return []  # valid list, zero clusters

    chunks = assign(props, cfg=None, group=fake_group)
    assert [c.text for c in chunks] == ["a", "b"]


def test_empty_title_split_marker_has_no_leading_space():
    props = [P(f"f{i}", "S", 0, 2) for i in range(3)]

    def fake_group(texts, cfg, max_props):
        return [{"proposition_indices": [0, 1, 2], "title": "", "summary": "s", "keywords": []}]

    chunks = assign(props, cfg=None, group=fake_group, window_size=100, max_props=1)
    assert [c.title for c in chunks] == ["(1/3)", "(2/3)", "(3/3)"]


def test_chunk_text_preserves_original_source_text_when_available():
    props = [
        Proposition("Cats purr.", 0, 27, "S", source_text="Cats purr. They also sleep."),
        Proposition("Cats sleep.", 0, 27, "S", source_text="Cats purr. They also sleep."),
    ]

    def fake_group(texts, cfg, max_props):
        return [{"proposition_indices": [0, 1], "title": "Cats", "summary": "s", "keywords": []}]

    chunks = assign(props, cfg=None, group=fake_group)

    assert chunks[0].text == "Cats purr. They also sleep."
    assert chunks[0].embedding_text == "Cats purr.\nCats sleep."
