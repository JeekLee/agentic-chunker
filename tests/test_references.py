from agentic_chunker._references import table_references


def test_table_references_ignore_appendix_labels() -> None:
    source = "별표 1 및 [별표3]은 제외하고 → 표 2와 [표 4]는 참조한다."

    assert table_references(source) == ["표 2", "표 4"]


def test_table_references_ignore_spaced_appendix_labels() -> None:
    source = "「의료급여법 시행령」[별 표1] 제3호 나목에 따른 기준이다."

    assert table_references(source) == []


def test_table_references_remove_caption_lines() -> None:
    source = "**[표 1]**\n\n본문에서 표 2를 비교한다."

    assert table_references(source) == ["표 2"]


def test_table_references_remove_angle_bracket_caption_lines() -> None:
    source = "<표 2> 의료급여비용 청구체계 <2021.1.1.삭제>\n본문이다."

    assert table_references(source) == []
