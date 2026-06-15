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
    """chat() then parse the reply as JSON. Strips ```json fences. None on failure."""
    text = chat(prompt, cfg)
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```\s*$", "", text)
    try:
        return json.loads(text)
    except Exception as exc:
        sys.stderr.write(f"  chat_json parse failed: {exc}\n")
        return None
