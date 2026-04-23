# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # Streaming Exploration — Does `tools` Kill Token-Level Streaming?
# MAGIC
# MAGIC ## Background
# MAGIC
# MAGIC When calling Databricks Model Serving endpoints with the Chat Completions API:
# MAGIC - `stream=true` **without tools** → token-level SSE chunks (great UX)
# MAGIC - `stream=true` **with tools** → entire response in ONE chunk (no typewriter effect)
# MAGIC
# MAGIC This was confirmed locally via raw HTTP tests against `databricks-llama-4-maverick`.
# MAGIC The collapse happens at the **endpoint level**, not in any Python client.
# MAGIC
# MAGIC ## Goal
# MAGIC
# MAGIC Test multiple models to answer:
# MAGIC 1. Is this a model-specific behavior or a serving infrastructure behavior?
# MAGIC 2. Do pay-per-token models (GPT-4o, GPT-4.1) behave differently?
# MAGIC 3. Does `tool_choice="none"` (hint: don't call tools) restore streaming?
# MAGIC
# MAGIC ## Models Tested
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────┬──────────────┬─────────────────┐
# MAGIC │ Model                               │ Type         │ Pricing         │
# MAGIC ├─────────────────────────────────────┼──────────────┼─────────────────┤
# MAGIC │ databricks-gpt-oss-120b             │ OSS (Llama)  │ DBU/token       │
# MAGIC │ databricks-llama-4-maverick         │ OSS (Llama4) │ DBU/token       │
# MAGIC │ databricks-claude-sonnet-4 (if avail)│ External     │ Pay-per-token   │
# MAGIC │ databricks-gpt-4.1 (if available)   │ External     │ Pay-per-token   │
# MAGIC │ databricks-gpt-4o (if available)    │ External     │ Pay-per-token   │
# MAGIC └─────────────────────────────────────┴──────────────┴─────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ## Test Matrix
# MAGIC
# MAGIC ```
# MAGIC For each model:
# MAGIC   ├── Test A: stream=true, NO tools        → expect many chunks
# MAGIC   ├── Test B: stream=true, WITH tools       → does it collapse?
# MAGIC   ├── Test C: stream=true, WITH tools,      → does tool_choice="none" help?
# MAGIC   │           tool_choice="none"
# MAGIC   └── Test D: stream=false (baseline)       → single response (control)
# MAGIC ```

# COMMAND ----------
# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# MAGIC %pip install openai databricks-sdk --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import json
import time
import urllib.request
from dataclasses import dataclass

# COMMAND ----------
# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC Edit this cell to add/remove models. Only models that exist as serving
# MAGIC endpoints in your workspace will be tested — others are skipped gracefully.

# COMMAND ----------

# Models to test — add or remove as needed
MODELS_TO_TEST = [
    "databricks-gpt-oss-120b",
    "databricks-llama-4-maverick",
    "databricks-claude-sonnet-4",
    "databricks-gpt-4.1",
    "databricks-gpt-4o",
]

PROMPT = "Tell me a short joke about programming."

# Tool definition (OpenAI Chat Completions format)
TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "search_docs",
        "description": "Search internal documentation",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
    },
}

# COMMAND ----------
# MAGIC %md
# MAGIC ## Auth & Endpoint Discovery
# MAGIC
# MAGIC We use the workspace-native token (available in notebooks) and check
# MAGIC which models actually exist before testing.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
HOST = w.config.host.rstrip("/")

# Get auth headers (works with both PAT and OAuth)
_auth = w.config.authenticate()
TOKEN = _auth.get("Authorization", "").removeprefix("Bearer ")

print(f"Host: {HOST}")
print(f"Token present: {bool(TOKEN)}")

# COMMAND ----------

# Discover which endpoints actually exist
available_endpoints = set()
for ep in w.serving_endpoints.list():
    available_endpoints.add(ep.name)

models_to_run = [m for m in MODELS_TO_TEST if m in available_endpoints]
models_skipped = [m for m in MODELS_TO_TEST if m not in available_endpoints]

print(f"Available endpoints to test: {models_to_run}")
if models_skipped:
    print(f"Skipped (not found): {models_skipped}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Test Functions
# MAGIC
# MAGIC Two layers of testing:
# MAGIC 1. **Raw HTTP** — bypasses all Python clients, tests the endpoint directly
# MAGIC 2. **OpenAI SDK** — tests via `openai.OpenAI()` which Databricks endpoints support
# MAGIC
# MAGIC Both use the same SSE (Server-Sent Events) streaming protocol.

# COMMAND ----------

@dataclass
class StreamResult:
    """Result of a single streaming test."""
    model: str
    test_name: str
    total_chunks: int
    content_chunks: int
    first_content_time: float   # seconds to first content chunk
    total_time: float           # seconds to [DONE]
    error: str = ""

    @property
    def streams(self) -> bool:
        """True if we got more than 1 content chunk (real streaming)."""
        return self.content_chunks > 2

    @property
    def label(self) -> str:
        return "STREAMS" if self.streams else "COLLAPSED"


def test_raw_http(model: str, test_name: str, include_tools: bool,
                  tool_choice: str | None = None) -> StreamResult:
    """
    Raw HTTP streaming test — no Python SDK, just urllib + SSE parsing.
    This is the ground truth: whatever happens here IS the endpoint behavior.
    """
    payload = {
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": True,
    }
    if include_tools:
        payload["tools"] = [TOOL_DEF]
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice

    url = f"{HOST}/serving-endpoints/{model}/invocations"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json",
        },
    )

    start = time.time()
    chunk_count = 0
    content_chunks = 0
    first_content_time = 0.0

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            for raw_line in resp:
                line = raw_line.decode().strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                chunk_count += 1
                try:
                    parsed = json.loads(data)
                    delta = parsed.get("choices", [{}])[0].get("delta", {})
                    if delta.get("content"):
                        content_chunks += 1
                        if content_chunks == 1:
                            first_content_time = time.time() - start
                except json.JSONDecodeError:
                    pass

        return StreamResult(
            model=model,
            test_name=test_name,
            total_chunks=chunk_count,
            content_chunks=content_chunks,
            first_content_time=first_content_time,
            total_time=time.time() - start,
        )
    except Exception as e:
        return StreamResult(
            model=model,
            test_name=test_name,
            total_chunks=0,
            content_chunks=0,
            first_content_time=0,
            total_time=time.time() - start,
            error=str(e)[:120],
        )

# COMMAND ----------

def test_openai_sdk(model: str, test_name: str, include_tools: bool,
                    tool_choice: str | None = None) -> StreamResult:
    """
    OpenAI SDK streaming test — uses the official OpenAI Python client
    pointed at the Databricks endpoint. This is how most apps will call it.
    """
    from openai import OpenAI

    client = OpenAI(
        base_url=f"{HOST}/serving-endpoints/{model}",
        api_key=TOKEN,
    )

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": True,
    }
    if include_tools:
        kwargs["tools"] = [TOOL_DEF]
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    start = time.time()
    chunk_count = 0
    content_chunks = 0
    first_content_time = 0.0

    try:
        stream = client.chat.completions.create(**kwargs)
        for chunk in stream:
            chunk_count += 1
            if chunk.choices and chunk.choices[0].delta.content:
                content_chunks += 1
                if content_chunks == 1:
                    first_content_time = time.time() - start

        return StreamResult(
            model=model,
            test_name=test_name,
            total_chunks=chunk_count,
            content_chunks=content_chunks,
            first_content_time=first_content_time,
            total_time=time.time() - start,
        )
    except Exception as e:
        return StreamResult(
            model=model,
            test_name=test_name,
            total_chunks=0,
            content_chunks=0,
            first_content_time=0,
            total_time=time.time() - start,
            error=str(e)[:120],
        )

# COMMAND ----------
# MAGIC %md
# MAGIC ## Run All Tests
# MAGIC
# MAGIC ```
# MAGIC For each model × each test variant → raw HTTP + OpenAI SDK
# MAGIC
# MAGIC Test variants:
# MAGIC   A: no tools           — baseline, should always stream
# MAGIC   B: with tools         — does it collapse?
# MAGIC   C: tools + choice=none — hint to not call tools, does streaming resume?
# MAGIC ```

# COMMAND ----------

all_results: list[StreamResult] = []

for model in models_to_run:
    print(f"\n{'='*70}")
    print(f"  MODEL: {model}")
    print(f"{'='*70}")

    tests = [
        ("A_no_tools",              False, None),
        ("B_with_tools",            True,  None),
        ("C_tools_choice_none",     True,  "none"),
    ]

    for test_name, tools, choice in tests:
        # --- Raw HTTP ---
        label = f"raw_{test_name}"
        print(f"  Running {label}...", end=" ", flush=True)
        r = test_raw_http(model, label, include_tools=tools, tool_choice=choice)
        all_results.append(r)
        if r.error:
            print(f"ERROR: {r.error[:60]}")
        else:
            print(f"{r.label} — {r.content_chunks} content chunks / {r.total_chunks} total  (TTFC={r.first_content_time:.2f}s)")

        # --- OpenAI SDK ---
        label = f"sdk_{test_name}"
        print(f"  Running {label}...", end=" ", flush=True)
        r = test_openai_sdk(model, label, include_tools=tools, tool_choice=choice)
        all_results.append(r)
        if r.error:
            print(f"ERROR: {r.error[:60]}")
        else:
            print(f"{r.label} — {r.content_chunks} content chunks / {r.total_chunks} total  (TTFC={r.first_content_time:.2f}s)")

print("\nAll tests complete.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Results Summary

# COMMAND ----------

# Build summary table
print(f"\n{'Model':<32} {'Test':<24} {'Chunks':>8} {'Content':>9} {'TTFC':>7} {'Total':>7} {'Status':<12}")
print("-" * 105)

for r in all_results:
    if r.error:
        status = f"ERROR"
    else:
        status = r.label

    ttfc = f"{r.first_content_time:.2f}s" if r.first_content_time > 0 else "—"
    total = f"{r.total_time:.2f}s" if r.total_time > 0 else "—"
    print(f"{r.model:<32} {r.test_name:<24} {r.total_chunks:>8} {r.content_chunks:>9} {ttfc:>7} {total:>7} {status:<12}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Diagnosis

# COMMAND ----------

# Group by model and analyze
from collections import defaultdict

by_model = defaultdict(dict)
for r in all_results:
    if not r.error:
        by_model[r.model][r.test_name] = r

print("=" * 70)
print("DIAGNOSIS PER MODEL")
print("=" * 70)

for model, tests in by_model.items():
    print(f"\n  {model}:")

    # Check raw HTTP results (ground truth)
    raw_a = tests.get("raw_A_no_tools")
    raw_b = tests.get("raw_B_with_tools")
    raw_c = tests.get("raw_C_tools_choice_none")

    if raw_a and raw_a.streams:
        print(f"    ✓ Streams without tools ({raw_a.content_chunks} chunks)")
    elif raw_a:
        print(f"    ✗ Does NOT stream even without tools — unusual")

    if raw_b and raw_b.streams:
        print(f"    ✓ Streams WITH tools ({raw_b.content_chunks} chunks) — GREAT!")
    elif raw_b:
        print(f"    ✗ Collapses with tools ({raw_b.content_chunks} chunk)")

    if raw_c and raw_c.streams:
        print(f"    ✓ tool_choice='none' RESTORES streaming ({raw_c.content_chunks} chunks)")
        print(f"      → WORKAROUND: set tool_choice='none' when tools are registered but not needed")
    elif raw_c:
        print(f"    ✗ tool_choice='none' does NOT help ({raw_c.content_chunks} chunk)")

    # Compare raw vs SDK
    sdk_b = tests.get("sdk_B_with_tools")
    if raw_b and sdk_b:
        if raw_b.streams == sdk_b.streams:
            print(f"    ≡ Raw HTTP and OpenAI SDK agree — behavior is endpoint-level")
        else:
            print(f"    ⚠ Raw HTTP and SDK differ — SDK may be introducing behavior")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Key Takeaways
# MAGIC
# MAGIC After running this notebook, you should know:
# MAGIC
# MAGIC 1. **Is streaming collapse model-specific?** → Compare OSS models vs pay-per-token models
# MAGIC 2. **Is it endpoint-level or client-level?** → Raw HTTP = endpoint truth. If both raw and SDK collapse, it's the endpoint.
# MAGIC 3. **Does `tool_choice="none"` help?** → If yes, agents can set this when routing decides "respond with text, don't call tools"
# MAGIC
# MAGIC ### Possible workarounds if collapse confirmed:
# MAGIC
# MAGIC | Option | How | Trade-off |
# MAGIC |--------|-----|-----------|
# MAGIC | `tool_choice="none"` | Set dynamically after routing decides no tool needed | Requires 2-pass: route first, then generate |
# MAGIC | Simulated streaming | Receive full text, yield char-by-char on frontend | Same TTFT, fake typewriter |
# MAGIC | Manual ReAct | Don't use `tools` param; parse tool calls from text | Fragile, model-dependent |
# MAGIC | Different model | Use a model that streams with tools (if any found above) | May affect quality/cost |

# COMMAND ----------
# MAGIC %md
# MAGIC ## Raw Chunk Inspector
# MAGIC
# MAGIC Pick a model and see exactly what the SSE stream looks like — useful for debugging
# MAGIC content format differences (plain string vs structured `[{type: reasoning}, {type: text}]`).

# COMMAND ----------

# Change this to inspect a specific model
INSPECT_MODEL = models_to_run[0] if models_to_run else "databricks-gpt-oss-120b"
INSPECT_WITH_TOOLS = True

print(f"Inspecting raw SSE stream: {INSPECT_MODEL} (tools={INSPECT_WITH_TOOLS})")
print("=" * 70)

payload = {
    "messages": [{"role": "user", "content": PROMPT}],
    "stream": True,
}
if INSPECT_WITH_TOOLS:
    payload["tools"] = [TOOL_DEF]

url = f"{HOST}/serving-endpoints/{INSPECT_MODEL}/invocations"
req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode(),
    headers={
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
    },
)

start = time.time()
with urllib.request.urlopen(req, timeout=60) as resp:
    for raw_line in resp:
        line = raw_line.decode().strip()
        if not line:
            continue
        if line.startswith("data: "):
            data = line[6:]
            elapsed = f"{time.time() - start:.3f}s"
            if data == "[DONE]":
                print(f"\n[{elapsed}] [DONE]")
                break
            try:
                parsed = json.loads(data)
                choice = parsed.get("choices", [{}])[0]
                delta = choice.get("delta", {})
                finish = choice.get("finish_reason")

                # Show the full delta for inspection
                print(f"\n[{elapsed}] finish={finish}")
                print(f"  delta keys: {list(delta.keys())}")
                if "content" in delta:
                    content = delta["content"]
                    if isinstance(content, str):
                        print(f"  content (str): {repr(content[:100])}")
                    elif isinstance(content, list):
                        print(f"  content (list, {len(content)} items):")
                        for item in content:
                            t = item.get("type", "?")
                            if t == "text":
                                print(f"    - text: {repr(item.get('text', '')[:80])}")
                            elif t == "reasoning":
                                summary = item.get("summary", [])
                                if summary:
                                    print(f"    - reasoning: {repr(summary[0].get('text', '')[:80])}")
                                else:
                                    print(f"    - reasoning (no summary)")
                            else:
                                print(f"    - {t}: {json.dumps(item)[:80]}")
                if "tool_calls" in delta:
                    print(f"  tool_calls: {json.dumps(delta['tool_calls'])[:120]}")
            except json.JSONDecodeError:
                print(f"\n[{elapsed}] RAW: {data[:120]}")
