"""OpenAI-compatible LLM client (stdlib urllib).

Mirrors md-converter's llm.py: one POST to /chat/completions, no external deps.
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from dataclasses import dataclass


@dataclass
class LlmConfig:
    url: str        # base URL, e.g. "http://localhost:10080/v1"
    api_key: str
    model: str      # e.g. "qwen3-30b-a3b"
    temperature: float = 0.0
    timeout: int = 120


def chat(prompt: str, cfg: LlmConfig) -> str:
    """Send a single user message, return the assistant text (or "" on failure)."""
    body = json.dumps({
        "model": cfg.model,
        "temperature": cfg.temperature,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    endpoint = f"{cfg.url.rstrip('/')}/chat/completions"
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
            result = json.loads(resp.read())
        return result["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        sys.stderr.write(f"  chat failed: {exc}\n")
        return ""


def chat_json(prompt: str, cfg: LlmConfig) -> object | None:
    """chat() then parse the reply as JSON.

    The model is asked for JSON-only responses, but OpenAI-compatible local
    models sometimes wrap valid JSON in prose or Markdown fences. Parse the
    strict response first, then fall back to the first balanced JSON object or
    array in the reply. Invalid JSON still returns ``None``.
    """
    text = chat(prompt, cfg)
    if not text:
        return None
    stripped = _strip_fence(text.strip())
    last_error: Exception | None = None
    for candidate in [stripped, *_json_candidates(stripped)]:
        try:
            return json.loads(candidate)
        except Exception as exc:
            last_error = exc
    sys.stderr.write(f"  chat_json parse failed: {last_error}\n")
    return None


def _strip_fence(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    return re.sub(r"\s*```\s*$", "", text)


def _json_candidates(text: str) -> list[str]:
    starts = [i for i, ch in enumerate(text) if ch in "[{"]
    candidates: list[str] = []
    for start in starts:
        candidate = _balanced_json_slice(text, start)
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _balanced_json_slice(text: str, start: int) -> str:
    pairs = {"[": "]", "{": "}"}
    if text[start] not in pairs:
        return ""

    stack = [pairs[text[start]]]
    in_string = False
    escaped = False
    for i in range(start + 1, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch in pairs:
            stack.append(pairs[ch])
        elif stack and ch == stack[-1]:
            stack.pop()
            if not stack:
                return text[start:i + 1]
    return ""
