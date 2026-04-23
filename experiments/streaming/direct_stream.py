"""
Direct streaming test — no server, no LangGraph overhead.
Tests each layer to validate token-level streaming.

Run: .venv/bin/python test_direct_stream.py [test_number]
  2 = ChatDatabricks astream (Chat Completions — default)
  5 = create_agent astream v1 (Chat Completions, no Responses API)
  6 = create_agent astream v2 (same, v2 format)
  all = run all tests
"""

import os
import sys
import time
import asyncio
from dotenv import load_dotenv


load_dotenv()

PROFILE = os.environ.get("DATABRICKS_CONFIG_PROFILE", "DEFAULT")
LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT", "databricks-gpt-oss-120b")
PROMPT = "Write me a haiku about the ocean."


def classify_content(content) -> tuple[str, str]:
    """Classify a chunk's content as reasoning, text, or empty.

    Returns (kind, display_text).
    """
    if not content:
        return ("empty", "")
    if isinstance(content, str):
        if content.startswith('[{"type":') or content.startswith('[{"type"'):
            return ("reasoning", content[:60])
        return ("text", content)
    if isinstance(content, list):
        # Raw list (not JSON-serialized) — check block types
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type", "unknown")
                if btype == "text":
                    return ("text", block.get("text", ""))
                elif btype == "reasoning":
                    summary = block.get("summary", [{}])
                    text = summary[0].get("text", "") if summary else ""
                    return ("reasoning", text[:60])
                elif btype == "refusal":
                    return ("refusal", block.get("refusal", ""))
        return ("unknown_list", str(content)[:60])
    return ("unknown", str(content)[:60])


def test_2_chat_databricks_default():
    """ChatDatabricks — astream() with default Chat Completions API."""
    print("=" * 60)
    print("TEST 2: ChatDatabricks — astream() [Chat Completions]")
    print("=" * 60)

    from databricks_langchain import ChatDatabricks

    llm = ChatDatabricks(endpoint=LLM_ENDPOINT)

    async def _run():
        start = time.time()
        counts = {"reasoning": 0, "text": 0, "empty": 0, "other": 0}
        async for chunk in llm.astream([{"role": "user", "content": PROMPT}]):
            elapsed = f"{time.time() - start:.2f}s"
            kind, display = classify_content(chunk.content)
            if kind == "text":
                counts["text"] += 1
                print(f"  [{elapsed}] TEXT: {repr(display)}")
            elif kind == "reasoning":
                counts["reasoning"] += 1
                print(f"  [{elapsed}] REASONING: {repr(display)}")
            elif kind == "empty":
                counts["empty"] += 1
            else:
                counts["other"] += 1
                print(f"  [{elapsed}] {kind}: {repr(display)}")

        print(f"\n  Counts: {counts}")

    asyncio.run(_run())
    print()


def test_5_create_agent_v1():
    """create_agent + ChatDatabricks (Chat Completions) — v1 format."""
    print("=" * 60)
    print("TEST 5: create_agent — astream v1 [Chat Completions]")
    print("  → stream_mode=['messages'] (v1 tuple format)")
    print("=" * 60)

    from databricks_langchain import ChatDatabricks
    from langchain.agents import create_agent

    llm = ChatDatabricks(endpoint=LLM_ENDPOINT)
    graph = create_agent(model=llm, tools=[], system_prompt="You are a helpful assistant.")

    async def _run():
        start = time.time()
        counts = {"reasoning": 0, "text": 0, "empty": 0, "other": 0}
        messages = {"messages": [{"role": "user", "content": PROMPT}]}

        async for event in graph.astream(input=messages, stream_mode=["messages"]):
            elapsed = f"{time.time() - start:.2f}s"
            # v1 list format: ("messages", (AIMessageChunk, metadata))
            mode = event[0]
            if mode != "messages":
                print(f"  [{elapsed}] UNEXPECTED mode: {mode}")
                continue

            chunk = event[1][0] if isinstance(event[1], tuple) else event[1]
            kind, display = classify_content(chunk.content)
            if kind == "text":
                counts["text"] += 1
                print(f"  [{elapsed}] TEXT: {repr(display)}")
            elif kind == "reasoning":
                counts["reasoning"] += 1
                print(f"  [{elapsed}] REASONING: {repr(display)}")
            elif kind == "empty":
                counts["empty"] += 1
            else:
                counts["other"] += 1
                print(f"  [{elapsed}] {kind}: {repr(display)}")

        print(f"\n  Counts: {counts}")

    asyncio.run(_run())
    print()


def test_6_create_agent_v2():
    """create_agent + ChatDatabricks (Chat Completions) — v2 format."""
    print("=" * 60)
    print("TEST 6: create_agent — astream v2 [Chat Completions]")
    print("  → stream_mode='messages', version='v2' (dict format)")
    print("=" * 60)

    from databricks_langchain import ChatDatabricks
    from langchain.agents import create_agent

    llm = ChatDatabricks(endpoint=LLM_ENDPOINT)
    graph = create_agent(model=llm, tools=[], system_prompt="You are a helpful assistant.")

    async def _run():
        start = time.time()
        counts = {"reasoning": 0, "text": 0, "empty": 0, "other": 0}
        messages = {"messages": [{"role": "user", "content": PROMPT}]}

        async for part in graph.astream(input=messages, stream_mode="messages", version="v2"):
            elapsed = f"{time.time() - start:.2f}s"
            # v2 format: {"type": "messages", "ns": (), "data": (AIMessageChunk, metadata)}
            if part["type"] != "messages":
                print(f"  [{elapsed}] UNEXPECTED type: {part['type']}")
                continue

            chunk, metadata = part["data"]
            kind, display = classify_content(chunk.content)
            if kind == "text":
                counts["text"] += 1
                print(f"  [{elapsed}] TEXT: {repr(display)}")
            elif kind == "reasoning":
                counts["reasoning"] += 1
                print(f"  [{elapsed}] REASONING: {repr(display)}")
            elif kind == "empty":
                counts["empty"] += 1
            else:
                counts["other"] += 1
                print(f"  [{elapsed}] {kind}: {repr(display)}")

        print(f"\n  Counts: {counts}")

    asyncio.run(_run())
    print()


TESTS = {
    "2": test_2_chat_databricks_default,
    "5": test_5_create_agent_v1,
    "6": test_6_create_agent_v2,
}

if __name__ == "__main__":
    print(f"Endpoint: {LLM_ENDPOINT}")
    print(f"Profile:  {PROFILE}")
    print(f"Prompt:   {PROMPT}")
    print()

    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which == "all":
        for test_fn in TESTS.values():
            try:
                test_fn()
            except Exception as e:
                print(f"  FAILED: {e}\n")
    elif which in TESTS:
        TESTS[which]()
    else:
        print(f"Unknown test: {which}. Use 2, 5, 6, or all.")
