"""
Quick ReAct streaming test — compares models end-to-end.

Builds a create_agent (ReAct) graph with one dummy tool, streams with astream,
and reports timestamp per chunk + max gap. If chunks arrive clustered in <50ms,
the endpoint is collapsing. If spread out over seconds, token streaming works.

Run: python test_react_stream.py
"""

import asyncio
import os
import time

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks
from langchain.agents import create_agent
from langchain_core.messages import AIMessageChunk, ToolMessage
from langchain_core.tools import tool

MODELS = [
    "databricks-meta-llama-3-3-70b-instruct",
    "databricks-gpt-oss-120b",
]

QUERY = "What's the weather in Tokyo? Answer in one sentence after calling the tool."


@tool
def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"The weather in {city} is sunny and 22°C."


async def run_model(model: str) -> dict:
    print(f"\n{'=' * 64}")
    print(f"MODEL: {model}")
    print(f"{'=' * 64}")

    workspace_client = WorkspaceClient(profile=os.environ.get("DATABRICKS_CONFIG_PROFILE", "DEFAULT"))

    llm = ChatDatabricks(
        endpoint=model,
        workspace_client=workspace_client,
    )
    graph = create_agent(llm, tools=[get_weather], system_prompt="You are helpful.")

    start = time.time()
    content_times = []
    tool_calls = 0
    tool_results = 0
    text_chunks = 0

    try:
        async for event in graph.astream(
            {"messages": [{"role": "user", "content": QUERY}]},
            stream_mode=["updates", "messages"],
        ):
            elapsed_ms = (time.time() - start) * 1000
            mode = event[0]
            data = event[1]

            if mode == "updates":
                for node_data in data.values():
                    for msg in node_data.get("messages", []):
                        if isinstance(msg, ToolMessage):
                            tool_results += 1
                            print(f"  [{elapsed_ms:7.1f}ms] ✅ TOOL_RESULT  {str(msg.content)[:60]!r}")
                        elif hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                tool_calls += 1
                                print(f"  [{elapsed_ms:7.1f}ms] 🔧 TOOL_CALL    {tc['name']}({tc.get('args', {})})")

            elif mode == "messages":
                chunk, metadata = data
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    content = chunk.content
                    if isinstance(content, list):
                        content = next(
                            (b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"),
                            "",
                        )
                    if content:
                        text_chunks += 1
                        content_times.append(elapsed_ms)
                        preview = content[:30] if isinstance(content, str) else str(content)[:30]
                        print(f"  [{elapsed_ms:7.1f}ms] 💬 TEXT_DELTA   {preview!r}")
    except Exception as e:
        print(f"  ❌ FAILED: {type(e).__name__}: {e}")
        return {"model": model, "error": str(e)}

    # Compute max gap between text chunks
    gaps = [content_times[i] - content_times[i - 1] for i in range(1, len(content_times))]
    max_gap = max(gaps) if gaps else 0
    mean_gap = sum(gaps) / len(gaps) if gaps else 0
    first_text_time = content_times[0] if content_times else None

    print("\n  Summary:")
    print(f"    tool_calls       : {tool_calls}")
    print(f"    tool_results     : {tool_results}")
    print(f"    text_chunks      : {text_chunks}")
    if first_text_time is not None:
        print(f"    first text at    : {first_text_time:.1f}ms")
    if gaps:
        print(f"    mean gap (text)  : {mean_gap:.1f}ms")
        print(f"    max gap (text)   : {max_gap:.1f}ms")
    # Decisive rule: multiple text chunks = real streaming, one chunk = collapse.
    # Gap timing is noisy for short responses (8 tokens at 15ms gap is still streaming).
    if text_chunks >= 2:
        print("    verdict          : ✅ STREAMING (token-level)")
    elif text_chunks == 1:
        print("    verdict          : ❌ SINGLE CHUNK (collapsed)")
    else:
        print("    verdict          : ⚠️  NO TEXT OUTPUT")

    return {
        "model": model,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "text_chunks": text_chunks,
        "mean_gap_ms": mean_gap,
        "max_gap_ms": max_gap,
    }


async def main() -> None:
    profile = os.environ.get("DATABRICKS_CONFIG_PROFILE", "DEFAULT")
    print(f"Profile: {profile}")
    print(f"Query:   {QUERY!r}")

    results = []
    for model in MODELS:
        results.append(await run_model(model))

    print(f"\n{'=' * 64}")
    print("FINAL COMPARISON")
    print(f"{'=' * 64}")
    print(f"{'model':<45} {'text_chunks':>12} {'mean_gap':>12}")
    print("-" * 72)
    for r in results:
        if "error" in r:
            print(f"{r['model']:<45} {'ERROR':>12} {'—':>12}")
        else:
            print(f"{r['model']:<45} {r['text_chunks']:>12} {r['mean_gap_ms']:>10.1f}ms")


if __name__ == "__main__":
    asyncio.run(main())
