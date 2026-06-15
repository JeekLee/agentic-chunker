"""Local end-to-end smoke test against a real OpenAI-compatible endpoint.

Run (credentials via env, never hardcoded):

    LLM_URL=http://localhost:10080/v1 \
    LLM_API_KEY=sk-... \
    LLM_MODEL=qwen3-... \
    .venv/bin/python examples/smoke_test.py

A manual integration check against a live endpoint — the test suite itself uses a
stub LLM and stays deterministic/offline. Credentials come from env vars only;
never hardcode a key here.
"""
import os
import sys

from agentic_chunker import AgenticChunker, LlmConfig

SAMPLE = """\
# 환경 정책 개요

탄소 중립은 2050년까지 온실가스 순배출량을 0으로 만드는 목표다.
정부는 이를 위해 재생에너지 비중을 늘리고 있다.

## 재생에너지

태양광 발전 설비는 지난 5년간 세 배로 증가했다.
풍력 발전도 해상 풍력을 중심으로 빠르게 확대되고 있다.

## 교통 부문

전기차 보급률은 신차 판매의 20%를 넘어섰다.
충전 인프라 확충이 핵심 과제로 남아 있다.
"""


def main() -> int:
    url = os.environ.get("LLM_URL")
    api_key = os.environ.get("LLM_API_KEY")
    model = os.environ.get("LLM_MODEL")
    if not (url and api_key and model):
        sys.stderr.write("Set LLM_URL, LLM_API_KEY, LLM_MODEL env vars.\n")
        return 2

    chunker = AgenticChunker(llm=LlmConfig(url=url, api_key=api_key, model=model))
    chunks = chunker.chunk(SAMPLE)

    print(f"\n=== {len(chunks)} chunk(s) ===\n")
    for c in chunks:
        print(f"[{c.index}] {c.title}")
        print(f"    keywords: {c.keywords}")
        print(f"    summary : {c.summary}")
        print(f"    spans   : {c.source_spans}")
        print(f"    text    : {c.text!r}\n")

    assert chunks, "expected at least one chunk"
    print("OK — end-to-end run produced chunks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
