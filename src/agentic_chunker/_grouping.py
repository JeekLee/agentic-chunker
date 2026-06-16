"""Agentic grouping over source-preserving evidence units."""
from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import json
import re

from ._models import Chunk
from .llm import LlmConfig
from .llm import chat_json as _real_chat_json

_MAX_CHUNK_SOURCE_CHARS = 6000
_TABLE_PAYLOAD_CHARS = 700
_TEXT_PAYLOAD_CHARS = 650
_TINY_SOURCE_CHARS = 20
_TEXTUAL_KINDS = {"text", "heading"}
_KEYWORD_RE = re.compile(r"[0-9A-Za-z가-힣]{2,}")
_SPACED_HANGUL_RE = re.compile(r"(?<![가-힣])[가-힣](?:\s+[가-힣])+(?![가-힣])")
_KEYWORD_STOPWORDS = {
    "그리고",
    "그러나",
    "대한",
    "따라",
    "또는",
    "및",
    "사항",
    "설명한다",
    "에서",
    "으로",
    "이를",
    "있는",
    "한다",
}

_PROMPT = """\
당신은 RAG 인덱싱용 청크 편집자입니다.
아래 evidence unit 목록을 의미적으로 일관된 청크로 묶어 주세요.

중요 원칙:
- unit의 원문 text는 절대 재작성하지 않습니다. unit id를 묶기만 합니다.
- 표를 참조하는 unit과 그 표 unit은 같은 답변 근거로 쓰일 가능성이 높으면 함께 묶습니다.
- 서로 다른 주제는 분리합니다.
- 한 청크에는 unit을 약 {max_units}개 이하로 담습니다.
- 모든 unit을 정확히 하나의 청크에 포함합니다.

Unit 목록:
{units}

다음 JSON 배열만 출력하세요:
[{{"unit_indices": [정수, ...], "title": "제목", "summary": "요약",
  "keywords": ["키워드", ...],
  "questions_answered": ["이 청크로 답변 가능한 질문", ...]}}, ...]
- unit_indices는 위 목록의 id입니다.
- title/summary/keywords/questions_answered는 원문과 표 맥락에 근거해 작성합니다.
- questions_answered는 2~3개만 작성하세요.
- JSON 문자열 안의 따옴표와 줄바꿈은 반드시 escape 하세요.
- 원문에 없는 사실은 추가하지 마세요.
"""

_ENRICH_PROMPT = """\
당신은 RAG 인덱싱용 청크 메타데이터 작성자입니다.
아래 청크 원문만 근거로 summary, keywords, questions_answered를 작성하세요.

규칙:
- source를 재작성하지 마세요.
- summary는 1~2문장으로 작성하세요.
- keywords는 검색에 유용한 3~8개 문자열로 작성하세요.
- questions_answered는 이 청크만 보고 답할 수 있는 질문 2~3개로 작성하세요.
- 원문에 없는 사실은 추가하지 마세요.

다음 JSON 객체만 출력하세요:
{{"summary": "요약", "keywords": ["키워드", ...],
  "questions_answered": ["질문", ...]}}

입력:
{payload}
"""

_QUESTIONS_RETRY_PROMPT = """\
아래 청크 원문만 근거로 이 청크만 보고 답할 수 있는 질문을 정확히 3개 작성하세요.
원문에 없는 사실은 추가하지 마세요.

다음 JSON 객체만 출력하세요:
{{"questions_answered": ["질문1", "질문2", "질문3"]}}

입력:
{payload}
"""


def group_units(
    units: list[Chunk],
    cfg: LlmConfig | None,
    *,
    group: Callable[[list[Chunk], LlmConfig | None, int], list | None] = None,
    max_units: int = 10,
    window_size: int = 40,
    concurrency: int = 8,
) -> list[Chunk]:
    """Group source-preserving units into final chunks."""
    if not units:
        return []
    group_fn = group or _default_group
    windows = _windows(units, window_size)

    def run(window: list[Chunk]) -> list[dict]:
        return _group_window(window, cfg, group_fn, max_units)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        per_window = list(ex.map(run, windows))

    chunk_dicts: list[dict] = []
    for window_chunks in per_window:
        for cd in window_chunks:
            chunk_dicts.append(cd)
    chunk_dicts = _merge_tiny_chunk_dicts(chunk_dicts, max_units)

    chunks: list[Chunk] = []
    for cd in chunk_dicts:
        chunk = _final_chunk(len(chunks), cd["units"], cd)
        chunks.append(chunk)
    _enrich_missing_metadata(chunks, cfg, concurrency)
    return chunks


def _default_group(units: list[Chunk], cfg: LlmConfig | None, max_units: int):
    payload = [_unit_payload(i, unit) for i, unit in enumerate(units)]
    prompt = _PROMPT.replace("{max_units}", str(max_units)).replace(
        "{units}",
        json.dumps(payload, ensure_ascii=False, indent=2),
    )
    raw = _real_chat_json(prompt, cfg)
    return raw if isinstance(raw, list) else None


def _windows(units: list[Chunk], window_size: int) -> list[list[Chunk]]:
    size = max(1, window_size)
    return [units[i:i + size] for i in range(0, len(units), size)]


def _unit_payload(local_id: int, unit: Chunk) -> dict:
    common = unit.metadata.get("common", {})
    table = unit.metadata.get("table", {})
    return {
        "id": local_id,
        "global_index": unit.index,
        "kind": common.get("chunk_kind", "text"),
        "section_path": common.get("section_path", []),
        "title": unit.title,
        "summary": _truncate(unit.summary, 350),
        "keywords": unit.keywords[:12],
        "table": _compact_table(table),
        "content": _payload_content(unit, common),
    }


def _payload_content(unit: Chunk, common: dict) -> str:
    kind = common.get("chunk_kind", "text")
    if kind in {"table", "table_part", "table_caption"}:
        return _truncate(unit.embedding_text or unit.text, _TABLE_PAYLOAD_CHARS)
    return _truncate(unit.text or unit.embedding_text, _TEXT_PAYLOAD_CHARS)


def _compact_table(table: dict) -> dict:
    if not table:
        return {}
    keys = ("table_id", "headers", "row_count", "part_index", "part_total")
    compact = {key: table[key] for key in keys if key in table and table[key] is not None}
    headers = compact.get("headers")
    if isinstance(headers, list):
        compact["headers"] = [str(header)[:80] for header in headers[:12]]
    return compact


def _group_window(
    window: list[Chunk],
    cfg: LlmConfig | None,
    group: Callable[[list[Chunk], LlmConfig | None, int], list | None],
    max_units: int,
) -> list[dict]:
    clusters = group(window, cfg, max_units)
    if not isinstance(clusters, list):
        return _fallback_chunks(window, max_units)

    chunk_dicts: list[dict] = []
    assigned: set[int] = set()
    for cl in clusters:
        if not isinstance(cl, dict):
            continue
        idxs = cl.get("unit_indices")
        if not isinstance(idxs, list):
            continue
        members: list[Chunk] = []
        for i in idxs:
            if isinstance(i, int) and 0 <= i < len(window) and i not in assigned:
                assigned.add(i)
                members.append(window[i])
        if not members:
            continue

        title = cl.get("title") if isinstance(cl.get("title"), str) else ""
        summary = cl.get("summary") if isinstance(cl.get("summary"), str) else ""
        embedding_text = cl.get("embedding_text") if isinstance(cl.get("embedding_text"), str) else ""
        kw = cl.get("keywords")
        keywords = [k for k in kw if isinstance(k, str)] if isinstance(kw, list) else []
        qa = cl.get("questions_answered")
        questions_answered = [q for q in qa if isinstance(q, str)] if isinstance(qa, list) else []

        capped = max(1, max_units)
        parts = _split_members(members, max_units=capped, max_source_chars=_MAX_CHUNK_SOURCE_CHARS)
        n = len(parts)
        for k, part in enumerate(parts, start=1):
            part_title = title
            if n > 1:
                part_title = f"{title} ({k}/{n})" if title else f"({k}/{n})"
            chunk_dicts.append({
                "units": part,
                "title": part_title,
                "summary": summary,
                "keywords": keywords,
                "questions_answered": questions_answered[:3],
                "embedding_text": embedding_text,
                "llm_metadata": True,
            })

    leftover = [window[i] for i in range(len(window)) if i not in assigned]
    chunk_dicts.extend(_fallback_chunks(leftover, max_units))
    return chunk_dicts or _fallback_chunks(window, max_units)


def _fallback_chunks(units: list[Chunk], max_units: int) -> list[dict]:
    groups: list[list[Chunk]] = []
    current: list[Chunk] = []

    def flush() -> None:
        nonlocal current
        if current:
            groups.append(current)
            current = []

    for unit in units:
        if not _is_textual_unit(unit):
            flush()
            groups.append([unit])
            continue

        if not current:
            current = [unit]
            continue

        if _can_merge_fallback(current, unit, max_units):
            current.append(unit)
        else:
            flush()
            current = [unit]

    flush()
    groups = _merge_tiny_fallback_groups(groups, max_units)
    return [_own_chunk(group) for group in groups]


def _merge_tiny_fallback_groups(groups: list[list[Chunk]], max_units: int) -> list[list[Chunk]]:
    merged: list[list[Chunk]] = []
    i = 0
    while i < len(groups):
        group = groups[i]
        if not _is_tiny_group(group):
            merged.append(group)
            i += 1
            continue

        if i + 1 < len(groups) and _can_merge_groups(group, groups[i + 1], max_units):
            groups[i + 1] = [*group, *groups[i + 1]]
            i += 1
            continue

        if merged and _can_merge_groups(merged[-1], group, max_units):
            merged[-1] = [*merged[-1], *group]
        else:
            merged.append(group)
        i += 1
    return merged


def _merge_tiny_chunk_dicts(items: list[dict], max_units: int) -> list[dict]:
    merged: list[dict] = []
    i = 0
    while i < len(items):
        item = items[i]
        units = item["units"]
        if not _is_tiny_group(units):
            merged.append(item)
            i += 1
            continue

        if i + 1 < len(items) and _can_merge_chunk_dicts(item, items[i + 1], max_units):
            items[i + 1] = _merge_chunk_dicts(item, items[i + 1])
            i += 1
            continue

        if merged and _can_merge_chunk_dicts(merged[-1], item, max_units):
            merged[-1] = _merge_chunk_dicts(merged[-1], item)
        else:
            merged.append(item)
        i += 1
    return merged


def _can_merge_chunk_dicts(left: dict, right: dict, max_units: int) -> bool:
    if not (_is_tiny_group(left["units"]) or _is_tiny_group(right["units"])):
        return False
    return _can_merge_groups(left["units"], right["units"], max_units)


def _merge_chunk_dicts(left: dict, right: dict) -> dict:
    units = [*left["units"], *right["units"]]
    if not left.get("llm_metadata") and not right.get("llm_metadata"):
        return _own_chunk(units)

    fallback = _own_chunk(units)
    primary = right if _is_tiny_group(left["units"]) else left
    keywords = _merge_list_fields([left.get("keywords", []), right.get("keywords", [])], 20) or fallback["keywords"]
    questions = _merge_list_fields(
        [left.get("questions_answered", []), right.get("questions_answered", [])],
        3,
    ) or fallback["questions_answered"]
    return {
        "units": units,
        "title": primary.get("title") or fallback["title"],
        "summary": _merge_text_fields([left.get("summary", ""), right.get("summary", "")], fallback["summary"]),
        "keywords": keywords,
        "questions_answered": questions,
        "embedding_text": "",
        "llm_metadata": bool(left.get("llm_metadata") or right.get("llm_metadata")),
    }


def _merge_text_fields(values: list[str], fallback: str) -> str:
    parts: list[str] = []
    for value in values:
        if value and value not in parts:
            parts.append(value)
    return " / ".join(parts)[:500] or fallback


def _merge_list_fields(values: list[list[str]], limit: int) -> list[str]:
    merged: list[str] = []
    for items in values:
        for item in items:
            if item and item not in merged:
                merged.append(item)
            if len(merged) >= limit:
                return merged
    return merged


def _is_tiny_group(group: list[Chunk]) -> bool:
    return len(_group_source(group).strip()) < _TINY_SOURCE_CHARS


def _can_merge_groups(left: list[Chunk], right: list[Chunk], max_units: int) -> bool:
    if len(left) + len(right) > _merge_unit_limit(left, right, max_units):
        return False
    if len(_group_source([*left, *right])) > _MAX_CHUNK_SOURCE_CHARS:
        return False
    return _compatible_sections(left, right)


def _merge_unit_limit(left: list[Chunk], right: list[Chunk], max_units: int) -> int:
    limit = max(1, max_units)
    if _is_tiny_group(left):
        return limit + len(left)
    if _is_tiny_group(right):
        return limit + len(right)
    return limit


def _group_source(group: list[Chunk]) -> str:
    return "\n\n".join(unit.source for unit in group if unit.source)


def _compatible_sections(left: list[Chunk], right: list[Chunk]) -> bool:
    left_sections = _group_section_keys(left)
    right_sections = _group_section_keys(right)
    return not left_sections or not right_sections or bool(left_sections & right_sections)


def _group_section_keys(group: list[Chunk]) -> set[tuple[str, ...]]:
    keys: set[tuple[str, ...]] = set()
    for unit in group:
        section_path = unit.metadata.get("common", {}).get("section_path", [])
        if section_path:
            keys.add(tuple(str(section) for section in section_path))
    return keys


def _can_merge_fallback(current: list[Chunk], unit: Chunk, max_units: int) -> bool:
    if len(current) >= max(1, max_units):
        return False
    if not all(_is_textual_unit(item) for item in current):
        return False
    if not _same_section(current[-1], unit):
        return False
    if sum(len(item.source) for item in current) + len(unit.source) > _MAX_CHUNK_SOURCE_CHARS:
        return False
    return any(_is_tiny_unit(item) for item in current) or _is_tiny_unit(unit)


def _same_section(left: Chunk, right: Chunk) -> bool:
    left_path = left.metadata.get("common", {}).get("section_path", [])
    right_path = right.metadata.get("common", {}).get("section_path", [])
    return left_path == right_path


def _is_textual_unit(unit: Chunk) -> bool:
    return unit.metadata.get("common", {}).get("chunk_kind", "text") in _TEXTUAL_KINDS


def _is_tiny_unit(unit: Chunk) -> bool:
    return len(unit.source.strip()) < _TINY_SOURCE_CHARS


def _own_chunk(units: list[Chunk]) -> dict:
    summary = _fallback_summary(units)
    keywords = _fallback_keywords(units)
    title = units[0].title
    return {
        "units": units,
        "title": title,
        "summary": summary,
        "keywords": keywords,
        "questions_answered": _fallback_questions(units, title),
        "embedding_text": _embedding_text(units, title, summary, keywords),
        "llm_metadata": False,
    }


def _split_members(
    members: list[Chunk],
    *,
    max_units: int,
    max_source_chars: int,
) -> list[list[Chunk]]:
    parts: list[list[Chunk]] = []
    current: list[Chunk] = []
    current_chars = 0

    for member in members:
        member_chars = len(member.source)
        would_exceed_units = len(current) >= max_units
        would_exceed_chars = current and current_chars + member_chars > max_source_chars
        if would_exceed_units or would_exceed_chars:
            parts.append(current)
            current = []
            current_chars = 0
        current.append(member)
        current_chars += member_chars

    if current:
        parts.append(current)
    return parts


def _final_chunk(index: int, units: list[Chunk], cd: dict) -> Chunk:
    ordered = sorted(units, key=lambda u: min((s[0] for s in u.source_spans), default=0))
    text = "\n\n".join(unit.text for unit in ordered if unit.text)
    source_spans = _dedupe_spans(ordered)
    title = cd.get("title") or _fallback_title(ordered)
    summary = cd.get("summary") or _fallback_summary(ordered)
    keywords = cd.get("keywords") or _fallback_keywords(ordered)
    questions_answered = cd.get("questions_answered") or _fallback_questions(ordered, title)
    embedding_text = cd.get("embedding_text") or _embedding_text(ordered, title, summary, keywords)
    metadata = _merged_metadata(ordered)
    metadata["_llm_metadata_generated"] = bool(cd.get("llm_metadata"))
    return Chunk(
        index=index,
        text=text,
        title=title,
        summary=summary,
        keywords=keywords,
        questions_answered=questions_answered,
        source_spans=source_spans,
        embedding_text=embedding_text,
        metadata=metadata,
    )


def _merged_metadata(units: list[Chunk]) -> dict:
    kinds: list[str] = []
    section_path: list[str] = []
    display_formats: list[str] = []
    unit_refs: list[dict] = []
    table_refs: list[dict] = []

    for unit in units:
        common = unit.metadata.get("common", {})
        kind = common.get("chunk_kind", "text")
        if kind not in kinds:
            kinds.append(kind)
        for section in common.get("section_path", []):
            if section not in section_path:
                section_path.append(section)
        fmt = common.get("display_format", "plain")
        if fmt not in display_formats:
            display_formats.append(fmt)

        table = unit.metadata.get("table", {})
        if table:
            table_refs.append({"unit_index": unit.index, **table})
        unit_refs.append({
            "unit_index": unit.index,
            "kind": kind,
            "table_id": table.get("table_id", ""),
        })

    chunk_kind = kinds[0] if len(kinds) == 1 else "mixed"
    display_format = display_formats[0] if len(display_formats) == 1 else "markdown"
    metadata = {
        "common": {
            "chunk_kind": chunk_kind,
            "unit_kinds": kinds,
            "section_path": section_path,
            "display_format": display_format,
        },
        "units": unit_refs,
    }
    if len(table_refs) == 1:
        metadata["table"] = {k: v for k, v in table_refs[0].items() if k != "unit_index"}
    elif table_refs:
        metadata["tables"] = table_refs
    return metadata


def _dedupe_spans(units: list[Chunk]) -> list[tuple[int, int]]:
    seen: list[tuple[int, int]] = []
    for unit in units:
        for span in unit.source_spans:
            normalized = tuple(span)
            if normalized not in seen:
                seen.append(normalized)
    return seen


def _fallback_title(units: list[Chunk]) -> str:
    return next((unit.title for unit in units if unit.title), units[0].text[:80] if units else "")


def _fallback_summary(units: list[Chunk]) -> str:
    return " / ".join(unit.summary for unit in units if unit.summary)[:500]


def _fallback_keywords(units: list[Chunk]) -> list[str]:
    keywords: list[str] = []
    for unit in units:
        for keyword in unit.keywords:
            if keyword not in keywords:
                keywords.append(keyword)
    if len(keywords) < 3:
        for keyword in _source_keywords(units):
            if keyword not in keywords:
                keywords.append(keyword)
            if len(keywords) >= 20:
                break
    return keywords[:20]


def _fallback_questions(units: list[Chunk], title: str) -> list[str]:
    questions: list[str] = []
    for unit in units:
        for question in unit.questions_answered:
            if question not in questions:
                questions.append(question)
    if len(questions) >= 2:
        return questions[:3]
    topic = title or _fallback_title(units)
    candidates = [
        f"{topic}에 대해 무엇을 알 수 있나요?" if topic else "이 청크에 대해 무엇을 알 수 있나요?",
        f"{topic}에서 확인해야 할 사항은 무엇인가요?" if topic else "이 청크에서 확인해야 할 사항은 무엇인가요?",
    ]
    for question in candidates:
        if question not in questions:
            questions.append(question)
        if len(questions) >= 2:
            break
    return questions[:3]


def _source_keywords(units: list[Chunk]) -> list[str]:
    counts: dict[str, int] = {}
    order: list[str] = []
    for unit in units:
        text = unit.text or unit.embedding_text or unit.summary or unit.title
        for keyword in _spaced_hangul_keywords(text):
            if keyword not in counts:
                order.append(keyword)
                counts[keyword] = 0
            counts[keyword] += 1
        for token in _KEYWORD_RE.findall(text):
            keyword = token.strip()
            if not _use_keyword(keyword):
                continue
            if keyword not in counts:
                order.append(keyword)
                counts[keyword] = 0
            counts[keyword] += 1
    return sorted(order, key=lambda keyword: (-counts[keyword], order.index(keyword)))[:20]


def _spaced_hangul_keywords(text: str) -> list[str]:
    keywords: list[str] = []
    for match in _SPACED_HANGUL_RE.finditer(text):
        keyword = re.sub(r"\s+", "", match.group(0))
        if _use_keyword(keyword) and keyword not in keywords:
            keywords.append(keyword)
    return keywords


def _use_keyword(keyword: str) -> bool:
    if keyword.lower() in _KEYWORD_STOPWORDS:
        return False
    if keyword.isdigit():
        return False
    if len(keyword) < 2:
        return False
    return True


def _embedding_text(units: list[Chunk], title: str, summary: str, keywords: list[str]) -> str:
    parts = []
    if title:
        parts.append(f"제목: {title}")
    if summary:
        parts.append(f"요약: {summary}")
    if keywords:
        parts.append("키워드: " + ", ".join(keywords))
    parts.extend(unit.embedding_text or unit.text for unit in units)
    return "\n".join(part for part in parts if part)


def _enrich_missing_metadata(chunks: list[Chunk], cfg: LlmConfig | None, concurrency: int) -> None:
    if cfg is None:
        return
    targets = [chunk for chunk in chunks if _needs_enrichment(chunk)]
    if not targets:
        return

    def run(chunk: Chunk) -> None:
        raw = _real_chat_json(_enrich_prompt(chunk), cfg)
        if not isinstance(raw, dict):
            return
        summary = raw.get("summary")
        if isinstance(summary, str) and summary.strip():
            chunk.summary = summary.strip()
        keywords = raw.get("keywords")
        if isinstance(keywords, list):
            cleaned = [k.strip() for k in keywords if isinstance(k, str) and k.strip()]
            if cleaned:
                chunk.keywords = cleaned[:12]
        questions = raw.get("questions_answered")
        if isinstance(questions, list):
            cleaned = [q.strip() for q in questions if isinstance(q, str) and q.strip()]
            if cleaned:
                chunk.questions_answered = cleaned[:3]
        chunk.metadata["_llm_metadata_generated"] = True
        chunk.embedding_text = _chunk_embedding_text(chunk)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        list(ex.map(run, targets))

    retry_targets = [chunk for chunk in targets if len(chunk.questions_answered) < 2]
    if retry_targets:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
            list(ex.map(lambda chunk: _retry_questions(chunk, cfg), retry_targets))
    for chunk in retry_targets:
        _ensure_min_questions(chunk)


def _needs_enrichment(chunk: Chunk) -> bool:
    return (
        not chunk.metadata.get("_llm_metadata_generated")
        or not chunk.summary
        or not chunk.keywords
        or len(chunk.questions_answered) < 2
    )


def _enrich_prompt(chunk: Chunk) -> str:
    payload = {
        "chunk_id": chunk.id,
        "source": _truncate(chunk.source, 5000),
        "current_summary": chunk.summary,
        "current_keywords": chunk.keywords,
        "current_questions_answered": chunk.questions_answered,
    }
    return _ENRICH_PROMPT.replace("{payload}", json.dumps(payload, ensure_ascii=False, indent=2))


def _chunk_embedding_text(chunk: Chunk) -> str:
    parts = [
        f"요약: {chunk.summary}" if chunk.summary else "",
        "키워드: " + ", ".join(chunk.keywords) if chunk.keywords else "",
        "답변 가능 질문: " + " / ".join(chunk.questions_answered) if chunk.questions_answered else "",
        chunk.embedding_text or chunk.source,
    ]
    return "\n".join(part for part in parts if part)


def _retry_questions(chunk: Chunk, cfg: LlmConfig) -> None:
    payload = {
        "chunk_id": chunk.id,
        "source": _truncate(chunk.source, 5000),
        "summary": chunk.summary,
        "keywords": chunk.keywords,
        "current_questions_answered": chunk.questions_answered,
    }
    prompt = _QUESTIONS_RETRY_PROMPT.replace("{payload}", json.dumps(payload, ensure_ascii=False, indent=2))
    raw = _real_chat_json(prompt, cfg)
    if not isinstance(raw, dict):
        return
    questions = raw.get("questions_answered")
    if not isinstance(questions, list):
        return
    cleaned = [q.strip() for q in questions if isinstance(q, str) and q.strip()]
    if cleaned:
        chunk.questions_answered = cleaned[:3]
        chunk.embedding_text = _chunk_embedding_text(chunk)


def _ensure_min_questions(chunk: Chunk) -> None:
    if len(chunk.questions_answered) >= 2:
        return
    topic = chunk.summary or chunk.title or next((line.strip() for line in chunk.source.splitlines() if line.strip()), "")
    candidates = [
        f"{topic}의 핵심 내용은 무엇인가요?" if topic else "이 청크의 핵심 내용은 무엇인가요?",
        f"{topic}에서 확인해야 할 사항은 무엇인가요?" if topic else "이 청크에서 확인해야 할 사항은 무엇인가요?",
    ]
    for question in candidates:
        if question not in chunk.questions_answered:
            chunk.questions_answered.append(question)
        if len(chunk.questions_answered) >= 2:
            break
    chunk.embedding_text = _chunk_embedding_text(chunk)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."
