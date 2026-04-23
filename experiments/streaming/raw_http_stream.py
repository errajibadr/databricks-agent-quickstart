"""
Raw HTTP streaming test — bypasses ChatDatabricks, LangChain, LangGraph.
Tests whether the Databricks endpoint itself streams with tools.
"""

import os
import time
import json
from dotenv import load_dotenv

load_dotenv()

from databricks.sdk import WorkspaceClient

PROFILE = os.environ.get("DATABRICKS_CONFIG_PROFILE", "DEFAULT")
LLM_ENDPOINT = "databricks-gpt-oss-120b"  # "databricks-meta-llama-3-3-70b-instruct"  # os.environ.get("LLM_ENDPOINT", "databricks-gpt-oss-120b")

w = WorkspaceClient(profile=PROFILE)
auth = w.config.authenticate()
token = auth["Authorization"].removeprefix("Bearer ")
host = w.config.host

url = f"{host}/serving-endpoints/{LLM_ENDPOINT}/invocations"

TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def test_stream(label, payload):
    print(f"\n{'=' * 60}")
    print(f"TEST: {label}")
    print(f"{'=' * 60}")

    import urllib.request

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    handler = urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(handler)

    start = time.time()
    chunk_count = 0
    with opener.open(req, timeout=60) as resp:
        print(f"  Status: {resp.status}")
        print(f"  Content-Type: {resp.headers.get('Content-Type')}")
        for raw_line in resp:
            line = raw_line.decode().strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    elapsed = f"{time.time() - start:.2f}s"
                    print(f"  [{elapsed}] [DONE]")
                    break
                chunk_count += 1
                elapsed = f"{time.time() - start:.2f}s"
                try:
                    parsed = json.loads(data)
                    delta = parsed.get("choices", [{}])[0].get("delta", {})
                    has_content = bool(delta.get("content"))
                    has_tool_calls = bool(delta.get("tool_calls"))
                    finish = parsed.get("choices", [{}])[0].get("finish_reason")
                    content_preview = ""
                    if has_content:
                        c = delta["content"]
                        if isinstance(c, str):
                            content_preview = repr(c[:50])
                        else:
                            content_preview = f"LIST len={len(c)}"
                    print(f"  [{elapsed}] #{chunk_count} content={has_content} tools={has_tool_calls} finish={finish} {content_preview}")
                except json.JSONDecodeError:
                    print(f"  [{elapsed}] #{chunk_count} RAW: {data[:80]}")

    print(f"  Total: {chunk_count} chunks")


# Test 1: No tools, stream
# test_stream(
#     "stream=true, NO tools",
#     {
#         "messages": [{"role": "user", "content": "tell me a joke"}],
#         "stream": True,
#     },
# )

# Test 2: With tools, stream
test_stream(
    "stream=true, WITH tools",
    {
        "messages": [{"role": "user", "content": "tell me a joke"}],
        "tools": [TOOL_DEF],
        "stream": True,
    },
)

test_stream(
    "stream=true, WITH tools",
    {
        "messages": [{"role": "user", "content": "What's the weather like in Tokyo?"}],
        "tools": [TOOL_DEF],
        "stream": True,
    },
)
