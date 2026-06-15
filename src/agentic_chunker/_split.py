"""Deterministic Markdown pre-split into blocks. No LLM.

Splits on ATX headers (# .. ######) to set the section, and on blank lines
into paragraph blocks within each section. Each block records its char
offsets into the original source and its nearest preceding header.
"""
from __future__ import annotations

import re

from ._common import Block

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def split(markdown: str) -> list[Block]:
    blocks: list[Block] = []
    header: str | None = None

    # Walk lines, tracking absolute char offsets. Accumulate non-blank,
    # non-header lines into a paragraph buffer; flush on blank line or header.
    buf: list[str] = []
    buf_start = 0
    pos = 0

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        text = "\n".join(buf).strip()
        if text:
            start = buf_start
            blocks.append(Block(text=text, char_start=start, char_end=start + len(text), header=header))
        buf = []

    for line in markdown.splitlines(keepends=True):
        stripped = line.strip()
        line_start = pos
        pos += len(line)

        m = _HEADER_RE.match(stripped)
        if m:
            flush()
            header = m.group(2).strip()
            continue
        if not stripped:
            flush()
            continue
        if not buf:
            buf_start = line_start + (len(line) - len(line.lstrip()))
        buf.append(line.rstrip("\r\n"))

    flush()
    return blocks
