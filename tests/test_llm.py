import json

import agentic_chunker.llm as llm_mod
from agentic_chunker.llm import LlmConfig, chat, chat_json


class _FakeResp:
    def __init__(self, payload: dict):
        self._data = json.dumps(payload).encode()

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch_reply(monkeypatch, content: str, capture: dict | None = None):
    def fake_urlopen(req, timeout=0):
        if capture is not None:
            capture["url"] = req.full_url
            capture["body"] = json.loads(req.data.decode())
            capture["headers"] = req.headers
        return _FakeResp({"choices": [{"message": {"content": content}}]})

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", fake_urlopen)


CFG = LlmConfig(url="http://localhost:10080/v1", api_key="k", model="m")


def test_chat_returns_message_content(monkeypatch):
    _patch_reply(monkeypatch, "hello world")
    assert chat("hi", CFG) == "hello world"


def test_chat_posts_to_chat_completions_with_auth(monkeypatch):
    cap = {}
    _patch_reply(monkeypatch, "ok", cap)
    chat("hi", CFG)
    assert cap["url"] == "http://localhost:10080/v1/chat/completions"
    assert cap["body"]["model"] == "m"
    assert cap["body"]["messages"][0]["content"] == "hi"
    assert cap["headers"]["Authorization"] == "Bearer k"


def test_chat_json_parses_plain_json(monkeypatch):
    _patch_reply(monkeypatch, '[{"a": 1}]')
    assert chat_json("give json", CFG) == [{"a": 1}]


def test_chat_json_strips_code_fence(monkeypatch):
    _patch_reply(monkeypatch, '```json\n{"a": 1}\n```')
    assert chat_json("give json", CFG) == {"a": 1}


def test_chat_json_returns_none_on_garbage(monkeypatch):
    _patch_reply(monkeypatch, "not json at all")
    assert chat_json("give json", CFG) is None
