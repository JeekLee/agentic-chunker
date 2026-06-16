"""Generic-ish source reference detection helpers.

The graph builder should not own source-text pattern details. This module keeps
those extraction rules isolated so additional table/figure/code reference
patterns can be added without changing graph construction.
"""
from __future__ import annotations

import re

_CAPTION_LINE_RE = re.compile(r"^\s*\**\s*\[?\s*표\s*\d+\s*\]?\s*\**\s*$", re.MULTILINE)
_TABLE_REF_RE = re.compile(r"(?:\[?\s*표\s*(\d+)\s*\]?|→\s*표\s*(\d+))")


def table_references(source: str) -> list[str]:
    """Return unique Korean table labels referenced by display source text."""
    text = _CAPTION_LINE_RE.sub("", source)
    refs: list[str] = []
    for match in _TABLE_REF_RE.finditer(text):
        num = match.group(1) or match.group(2)
        ref = f"표 {num}"
        if ref not in refs:
            refs.append(ref)
    return refs
