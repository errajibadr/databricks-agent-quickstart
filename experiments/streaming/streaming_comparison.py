"""
Streaming comparison: ChatDatabricks vs ChatOpenAI vs Raw HTTP
All hitting the SAME Databricks endpoint, with and without tools.

Goal: isolate whether streaming collapse is:
  A) ChatDatabricks client-side bug (bind_tools implementation)
  B) Databricks endpoint behavior (Chat Completions API with tools)
  C) Model-specific (llama-4-maverick doesn't stream with tools)
"""

import asyncio
import json
import os
import time
import urllib.request

from dotenv import load_dotenv

load_dotenv(override=True)

from databricks.sdk import WorkspaceClient

# --- Auth (works with databricks-cli OAuth, not just static PATs) ---
# Force DEFAULT profile — shell may have a stale DATABRICKS_CONFIG_PROFILE
PROFILE = "DEFAULT"
LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT", "databricks-gpt-oss-120b")

w = WorkspaceClient(profile=PROFILE)
HOST = w.config.host.rstrip("/")
# authenticate() returns {"Authorization": "Bearer <jwt>"} — works with OAuth U2M
AUTH_HEADERS = w.config.authenticate()
TOKEN = AUTH_HEADERS["Authorization"].removeprefix("Bearer ")

print(f"Host: {HOST}")
print(f"Endpoint: {LLM_ENDPOINT}")
print(f"Token present: {bool(TOKEN)}")
print(f"Token prefix: {TOKEN[:30]}...")

# --- Shared tool definition ---
TOOL_OPENAI_FORMAT = {
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

PROMPT = "Tell me a short joke."


# ============================================================
# TEST 1: Raw HTTP — no Python client at all
# ============================================================
def test_raw_http(label, include_tools):
    print(f"\n{'=' * 60}")
    print(f"TEST: Raw HTTP — {label}")
    print(f"{'=' * 60}")

    payload = {
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": True,
    }
    if include_tools:
        payload["tools"] = [TOOL_OPENAI_FORMAT]

    url = f"{HOST}/serving-endpoints/{LLM_ENDPOINT}/invocations"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json",
        },
    )
    # Skip proxy to avoid SOCKS issues in sandbox
    handler = urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(handler)

    start = time.time()
    chunk_count = 0
    content_chunks = 0

    with opener.open(req, timeout=60) as resp:
        print(f"  Status: {resp.status}")
        for raw_line in resp:
            line = raw_line.decode().strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    print(f"  [{time.time() - start:.2f}s] [DONE]")
                    break
                chunk_count += 1
                try:
                    parsed = json.loads(data)
                    delta = parsed.get("choices", [{}])[0].get("delta", {})
                    has_content = bool(delta.get("content"))
                    has_tool_calls = bool(delta.get("tool_calls"))
                    finish = parsed.get("choices", [{}])[0].get("finish_reason")
                    if has_content:
                        content_chunks += 1
                        c = delta["content"]
                        preview = repr(c[:60]) if isinstance(c, str) else f"LIST({len(c)} items)"
                        print(f"  [{time.time() - start:.2f}s] #{chunk_count} CONTENT {preview}")
                    elif has_tool_calls:
                        print(f"  [{time.time() - start:.2f}s] #{chunk_count} TOOL_CALL")
                    elif finish:
                        print(f"  [{time.time() - start:.2f}s] #{chunk_count} finish={finish}")
                    # Skip empty deltas (role-only, etc.)
                except json.JSONDecodeError:
                    print(f"  [{time.time() - start:.2f}s] #{chunk_count} RAW: {data[:80]}")

    print(f"\n  RESULT: {chunk_count} total chunks, {content_chunks} with content")
    return content_chunks


# ============================================================
# TEST 2: ChatDatabricks (what we've been using)
# ============================================================
async def test_chat_databricks(label, include_tools):
    from databricks_langchain import ChatDatabricks
    from langchain_core.tools import tool

    @tool
    def get_weather(city: str) -> str:
        """Get weather for a city"""
        return "sunny"

    print(f"\n{'=' * 60}")
    print(f"TEST: ChatDatabricks — {label}")
    print(f"{'=' * 60}")

    llm = ChatDatabricks(endpoint=LLM_ENDPOINT)
    if include_tools:
        llm = llm.bind_tools([get_weather])

    start = time.time()
    chunk_count = 0
    content_chunks = 0

    async for chunk in llm.astream(PROMPT):
        chunk_count += 1
        has = bool(getattr(chunk, "content", None))
        if has:
            content_chunks += 1
            c = chunk.content
            preview = repr(c[:60]) if isinstance(c, str) else f"type={type(c).__name__}"
            print(f"  [{time.time() - start:.2f}s] #{chunk_count} CONTENT {preview}")

    print(f"\n  RESULT: {chunk_count} total chunks, {content_chunks} with content")
    return content_chunks


# ============================================================
# TEST 3: ChatOpenAI pointing at Databricks endpoint
# ============================================================
async def test_chat_openai(label, include_tools):
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool

    @tool
    def get_weather(city: str) -> str:
        """Get weather for a city"""
        return "sunny"

    print(f"\n{'=' * 60}")
    print(f"TEST: ChatOpenAI → Databricks — {label}")
    print(f"{'=' * 60}")

    # ChatOpenAI appends /chat/completions to base_url
    # Databricks expects: /serving-endpoints/{name}/invocations
    # So base_url must end at the endpoint root for the OpenAI client
    llm = ChatOpenAI(
        base_url=f"{HOST}/serving-endpoints/{LLM_ENDPOINT}/invocations/",
        api_key=TOKEN,
        model=LLM_ENDPOINT,
    )
    if include_tools:
        llm = llm.bind_tools([get_weather])

    start = time.time()
    chunk_count = 0
    content_chunks = 0

    async for chunk in llm.astream(PROMPT):
        chunk_count += 1
        has = bool(getattr(chunk, "content", None))
        if has:
            content_chunks += 1
            c = chunk.content
            preview = repr(c[:60]) if isinstance(c, str) else f"type={type(c).__name__}"
            print(f"  [{time.time() - start:.2f}s] #{chunk_count} CONTENT {preview}")

    print(f"\n  RESULT: {chunk_count} total chunks, {content_chunks} with content")
    return content_chunks


# ============================================================
# MAIN — run all tests
# ============================================================
if __name__ == "__main__":
    results = {}

    # Layer 1: Raw HTTP (ground truth)
    print("\n" + "#" * 60)
    print("# LAYER 1: RAW HTTP (no Python client)")
    print("#" * 60)
    results["raw_no_tools"] = test_raw_http("no tools", include_tools=False)
    results["raw_with_tools"] = test_raw_http("WITH tools", include_tools=True)

    # Layer 2: ChatDatabricks
    print("\n" + "#" * 60)
    print("# LAYER 2: ChatDatabricks")
    print("#" * 60)
    results["cdb_no_tools"] = asyncio.run(test_chat_databricks("no tools", include_tools=False))
    results["cdb_with_tools"] = asyncio.run(test_chat_databricks("WITH tools", include_tools=True))

    # Layer 3: ChatOpenAI
    print("\n" + "#" * 60)
    print("# LAYER 3: ChatOpenAI → same Databricks endpoint")
    print("#" * 60)
    results["openai_no_tools"] = asyncio.run(test_chat_openai("no tools", include_tools=False))
    results["openai_with_tools"] = asyncio.run(test_chat_openai("WITH tools", include_tools=True))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Test':<30} {'Content chunks':>15}")
    print("-" * 45)
    for k, v in results.items():
        print(f"  {k:<28} {v:>13}")
    print()

    # Diagnosis
    raw_ok = results["raw_no_tools"] > 2
    raw_tools_ok = results["raw_with_tools"] > 2
    cdb_ok = results["cdb_no_tools"] > 2
    cdb_tools_ok = results["cdb_with_tools"] > 2
    openai_ok = results["openai_no_tools"] > 2
    openai_tools_ok = results["openai_with_tools"] > 2

    print("DIAGNOSIS:")
    if not raw_tools_ok:
        print("  → Endpoint itself collapses with tools — NOT a client bug")
        print("  → This is a Chat Completions API / model limitation")
    elif raw_tools_ok and not cdb_tools_ok and openai_tools_ok:
        print("  → ChatDatabricks.bind_tools() is the culprit")
        print("  → WORKAROUND: use ChatOpenAI instead")
    elif raw_tools_ok and not cdb_tools_ok and not openai_tools_ok:
        print("  → Both clients collapse — likely an endpoint-level behavior")
        print("  → The raw HTTP test may show partial streaming the clients discard")
    elif raw_tools_ok and cdb_tools_ok and openai_tools_ok:
        print("  → Everything streams! Previous issue may have been transient")
    else:
        print("  → Unexpected pattern — investigate manually")
