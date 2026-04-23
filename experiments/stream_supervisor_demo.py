# Databricks notebook source
# MAGIC %md
# MAGIC # [EXPERIMENT] Supervisor Streaming — Responses API SSE
# MAGIC
# MAGIC **Status:** exploratory — lives under `experiments/` because it investigates
# MAGIC behavior we don't yet have a fully settled mental model for. Once observations
# MAGIC are stable and the event-parsing pattern is solidified, graduate into a
# MAGIC numbered pedagogical notebook (e.g. `10_chainlit_streaming.py`) that teaches
# MAGIC the pattern as a built-in quickstart step.
# MAGIC
# MAGIC ## Purpose
# MAGIC
# MAGIC Test what events an Agent Bricks Supervisor (or any Responses-API-compatible
# MAGIC serving endpoint) emits when called with `stream=True`. Serves as:
# MAGIC
# MAGIC 1. A **teaching prop** for the V1 / V2 streaming conversation — concrete
# MAGIC    observable events make the abstract discussion tractable.
# MAGIC 2. A **debugging tool** for unexpected event shapes (e.g. the
# MAGIC    `"expecting value line 1 col 1"` tool-extraction error seen during
# MAGIC    `mlflow.genai.evaluate()` runs).
# MAGIC 3. A **verification harness** for the "can Supervisor forward sub-agent
# MAGIC    stream events?" question (Q1 in `docs/dachser/diagrams/supervisor-streaming-prompt.md`).
# MAGIC
# MAGIC ## What we test
# MAGIC
# MAGIC 1. **Token-level streaming shape:** how often does `response.output_text.delta`
# MAGIC    fire? Is the delta a plain string or a list of blocks (reasoning vs text)?
# MAGIC    Some models (e.g. `gpt-oss-120b`) return structured content.
# MAGIC 2. **Tool dispatch visibility:** does `response.output_item.added`
# MAGIC    (`function_call`) fire before or after the Supervisor routes? At what latency?
# MAGIC 3. **Tool result pattern:** does `response.output_item.done`
# MAGIC    (`function_call_output`) arrive as a single block, or are inner tokens
# MAGIC    streamed if the sub-agent is a streaming ResponsesAgent? Answers the
# MAGIC    "can Supervisor forward sub-agent streams?" question.
# MAGIC 4. **Args streaming:** does `response.function_call_arguments.delta` fire
# MAGIC    incrementally, or does the Supervisor assemble args before emitting?
# MAGIC 5. **Long-task handling:** does `task_continue_request` fire for multi-source
# MAGIC    queries with `databricks_options.long_task: True`?
# MAGIC
# MAGIC ## Related
# MAGIC
# MAGIC - `streaming/exploration_notebook.py` — sibling experiment for **Chat Completions API**
# MAGIC   (Supervisor does NOT use Chat Completions; this is the Responses API counterpart).
# MAGIC - `streaming/README.md` — investigation write-up with model compatibility table.
# MAGIC - `03_agent.py` — the **emitting** side: how to produce these events from a
# MAGIC   custom ResponsesAgent (this notebook is the **consuming** side).
# MAGIC - `doc-agent-app/agent_server/agent.py` — Apps variant of the emitting side.
# MAGIC - `../dbx-agent-lab/docs/dachser/diagrams/supervisor-streaming-prompt.md` —
# MAGIC   conceptual architecture (three zones: client / `aroll` / sub-agents).
# MAGIC - DACHSER Session 7 (2026-04-20) — where V1 / V2 streaming was first framed.
# MAGIC
# MAGIC ## Event type reference
# MAGIC
# MAGIC | Event type | When it fires | Render as |
# MAGIC |---|---|---|
# MAGIC | `response.output_item.added` (function_call) | Supervisor decides to call a sub-agent | "Querying {tool}..." badge |
# MAGIC | `response.function_call_arguments.delta` | Tool args being constructed | Optional args preview |
# MAGIC | `response.output_item.done` (function_call) | Full tool call assembled | Finalize tool badge |
# MAGIC | `response.output_item.done` (function_call_output) | Sub-agent returned | "✓ Got result" or collapse |
# MAGIC | `response.output_text.delta` | Final-answer token | Append to message bubble |
# MAGIC | `response.output_item.done` (message) | Final answer complete | Finalize bubble |
# MAGIC | `task_continue_request` | Long task paused | Auto-resume (partial impl here) |

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Install
# MAGIC
# MAGIC `openai` is usually preinstalled on Databricks runtimes. Uncomment if needed.

# COMMAND ----------

# %pip install --quiet openai
# dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Configuration
# MAGIC
# MAGIC Pulls `SUPERVISOR_NAME` from `_config.py` — the repo's single source of truth.
# MAGIC Change the Supervisor endpoint name there, not here.
# MAGIC
# MAGIC To smoke-test streaming plumbing without a Supervisor, override `ENDPOINT_NAME`
# MAGIC below with a Responses-API-capable endpoint (e.g. `"databricks-gpt-oss-120b"`).

# COMMAND ----------
# MAGIC %run ../_config

# COMMAND ----------

ENDPOINT_NAME = SUPERVISOR_NAME   # from _config.py — the endpoint deployed by 07_supervisor.py
QUERY = (
    "What does LangChain offer for building RAG pipelines, "
    "and what projects are currently tracked in our project tracker?"
)

# Debug toggles
RAW_MODE = False   # True = dump raw event objects (useful for the eval extraction bug)
SHOW_ARGS_DELTAS = False   # True = print every function_call_arguments.delta chunk

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Auth + client
# MAGIC
# MAGIC In a Databricks notebook, `WorkspaceClient()` picks up the notebook's own auth.
# MAGIC `get_open_ai_client()` returns a correctly-configured OpenAI SDK client pointed
# MAGIC at the workspace's serving endpoints base URL.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
openai_client = w.serving_endpoints.get_open_ai_client()

print(f"Workspace: {w.config.host}")
print(f"Endpoint:  {ENDPOINT_NAME}")
print(f"Query:     {QUERY!r}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Event parser (inlined from `databricks-bot-service/teams-bot/client/stream_handler.py`)
# MAGIC
# MAGIC Maps raw Responses API events to small typed records. Keeps the streaming
# MAGIC loop below readable. Unknown event types fall through to `None` — mirrors
# MAGIC the Responses API's forward-compatibility guarantee.

# COMMAND ----------

from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class ContentDelta:
    text: str                   # a token chunk

@dataclass
class ToolCallDone:
    call_id: str
    name: str
    arguments: str              # full JSON string, already assembled

@dataclass
class ToolResultDone:
    call_id: str
    output: str

@dataclass
class ArgsDelta:
    call_id: str
    delta: str                  # partial args being streamed

@dataclass
class TaskContinueRequest:
    continue_id: str
    step: int

StreamEvent = Union[ContentDelta, ToolCallDone, ToolResultDone, ArgsDelta, TaskContinueRequest]


def parse_event(event) -> Optional[StreamEvent]:
    """Classify a raw Responses API event. Returns None for ignored events."""
    t = getattr(event, "type", None)

    if t == "response.output_text.delta":
        return ContentDelta(text=getattr(event, "delta", "") or "")

    if t == "response.function_call_arguments.delta":
        return ArgsDelta(
            call_id=getattr(event, "item_id", "") or getattr(event, "call_id", ""),
            delta=getattr(event, "delta", "") or "",
        )

    if t == "response.output_item.done":
        item = getattr(event, "item", None)
        if item is None:
            return None
        item_type = getattr(item, "type", None)

        if item_type == "function_call":
            return ToolCallDone(
                call_id=getattr(item, "call_id", ""),
                name=getattr(item, "name", ""),
                arguments=getattr(item, "arguments", "") or "",
            )
        if item_type == "function_call_output":
            return ToolResultDone(
                call_id=getattr(item, "call_id", ""),
                output=getattr(item, "output", "") or "",
            )
        if item_type == "task_continue_request":
            return TaskContinueRequest(
                continue_id=getattr(item, "id", ""),
                step=getattr(item, "step", 0),
            )
        # message, other item types: ignored at this layer
        return None

    # response.created / response.in_progress / response.completed / ... → ignored
    return None


# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Pretty streaming loop
# MAGIC
# MAGIC Calls the endpoint with `stream=True` and prints a timestamped, emoji-prefixed
# MAGIC line per event. First text delta is the **real TTFT** — the number Chainlit UX
# MAGIC will feel.

# COMMAND ----------

import time
import json


def stream_query(endpoint: str, query: str):
    input_messages = [{"role": "user", "content": query}]

    print(f"📨 Sending query to '{endpoint}'...")
    print(f"   {query!r}\n")

    start = time.time()
    ttft = None

    counts = {"tool_calls": 0, "tool_results": 0, "text_deltas": 0, "args_deltas": 0, "task_continue": 0}
    text_buffer = []

    # Sync Responses API call. The OpenAI SDK handles SSE parsing and yields
    # event objects with `.type` + type-specific fields.
    stream = openai_client.responses.create(
        model=endpoint,
        input=input_messages,
        stream=True,
    )

    print(f"[{time.time() - start:5.2f}s] 📡 STREAM OPENED\n")

    for raw in stream:
        if RAW_MODE:
            # Dump the raw event — useful for debugging unexpected shapes
            # (e.g. the "expecting value line 1 col 1" bug the DACHSER team is chasing)
            print(f"[{time.time() - start:5.2f}s] 🧪 RAW {getattr(raw, 'type', '?')}: {raw!r}"[:200])

        evt = parse_event(raw)
        if evt is None:
            continue

        elapsed = f"{time.time() - start:5.2f}s"

        if isinstance(evt, ToolCallDone):
            counts["tool_calls"] += 1
            try:
                args_preview = json.dumps(json.loads(evt.arguments))
            except Exception:
                args_preview = evt.arguments
            print(f"[{elapsed}] 🔧 TOOL_CALL          {evt.name}({args_preview[:120]})")

        elif isinstance(evt, ArgsDelta):
            counts["args_deltas"] += 1
            if SHOW_ARGS_DELTAS:
                print(f"[{elapsed}] 🧩 ARGS_DELTA        +{evt.delta!r}")

        elif isinstance(evt, ToolResultDone):
            counts["tool_results"] += 1
            output_preview = evt.output.replace("\n", " ")[:80]
            print(f"[{elapsed}] ✅ TOOL_RESULT        ({len(evt.output)} chars) {output_preview!r}")

        elif isinstance(evt, ContentDelta):
            counts["text_deltas"] += 1
            if ttft is None:
                ttft = time.time() - start
                print(f"[{elapsed}] ⏱  TTFT              (first text delta)")
            text_buffer.append(evt.text)
            print(f"[{elapsed}] 💬 TOKEN              {evt.text!r}")

        elif isinstance(evt, TaskContinueRequest):
            counts["task_continue"] += 1
            print(f"[{elapsed}] ⏸  TASK_CONTINUE      id={evt.continue_id} step={evt.step}")
            print(f"           ↳ In production, resume by sending a task_continue_response.")
            print(f"           ↳ AI Playground auto-handles this; Chainlit must implement.")

    total = time.time() - start
    full_text = "".join(text_buffer)

    print()
    print("═" * 56)
    print(" Summary")
    print("═" * 56)
    print(f"  TTFT (first text delta):    {ttft:.2f}s" if ttft else "  TTFT:                       — (no text deltas)")
    print(f"  Total time:                 {total:.2f}s")
    print(f"  Tool calls:                 {counts['tool_calls']}")
    print(f"  Tool results:               {counts['tool_results']}")
    print(f"  Text deltas:                {counts['text_deltas']}")
    print(f"  Args deltas:                {counts['args_deltas']} (SHOW_ARGS_DELTAS={SHOW_ARGS_DELTAS})")
    print(f"  task_continue_request:      {counts['task_continue']}")
    print(f"  Assembled response:         {len(full_text)} chars")
    print("═" * 56)
    if full_text:
        print("\n📝 FULL RESPONSE:\n")
        print(full_text)

    return full_text, counts


# COMMAND ----------

response_text, counts = stream_query(ENDPOINT_NAME, QUERY)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Debug mode — raw event dump
# MAGIC
# MAGIC Set `RAW_MODE = True` at the top and rerun Cell 5 if you need to see the
# MAGIC exact event shapes — useful for chasing the tool-extraction bug the DACHSER
# MAGIC team is working on, or for verifying whether a sub-agent's stream events are
# MAGIC being forwarded (the "can Supervisor forward sub-agent streams?" question).
# MAGIC
# MAGIC **Things to look for in raw mode:**
# MAGIC
# MAGIC 1. **Does `function_call.arguments` arrive as one final block, or as many
# MAGIC    `function_call_arguments.delta` events?** Some endpoints stream the args;
# MAGIC    Agent Bricks Supervisor typically assembles them first.
# MAGIC
# MAGIC 2. **Are sub-agent inner tokens visible?** If the Supervisor forwards
# MAGIC    sub-agent streaming, you'll see `output_text.delta` events *between*
# MAGIC    tool call and tool result, not just after. If you only see one big
# MAGIC    `function_call_output` block with the full sub-agent response inside,
# MAGIC    forwarding is NOT happening → V2 streaming would require a custom
# MAGIC    Supervisor. This is the structural question from DACHSER Session 7.
# MAGIC
# MAGIC 3. **Does `task_continue_request` fire?** On long multi-tool queries,
# MAGIC    enable `databricks_options: {long_task: true}` (see Cell 7 below) and
# MAGIC    watch for this event.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Long-task mode (optional)
# MAGIC
# MAGIC For multi-tool queries that may exceed the default request timeout,
# MAGIC the Supervisor supports a continuation protocol. Enable it via
# MAGIC `databricks_options` and handle the `task_continue_request` event by
# MAGIC sending a follow-up request with a `task_continue_response`.
# MAGIC
# MAGIC The OpenAI SDK doesn't natively know about `databricks_options`, so we
# MAGIC add it via `extra_body`.

# COMMAND ----------

# Example — try this with a query that should provoke multiple sub-agent calls
LONG_QUERY = (
    "Give me a comprehensive overview of what LangChain offers for: "
    "RAG pipelines, agent orchestration, streaming, and memory. "
    "Then cross-reference against active projects in our project tracker "
    "and identify which ones could benefit from each capability."
)

def stream_long_task(endpoint: str, query: str):
    input_messages = [{"role": "user", "content": query}]
    start = time.time()

    stream = openai_client.responses.create(
        model=endpoint,
        input=input_messages,
        stream=True,
        extra_body={"databricks_options": {"long_task": True}},
    )

    for raw in stream:
        evt = parse_event(raw)
        if evt is None:
            continue
        elapsed = f"{time.time() - start:5.2f}s"

        if isinstance(evt, TaskContinueRequest):
            print(f"[{elapsed}] ⏸  TASK_CONTINUE fired at step {evt.step}")
            print(f"           → id={evt.continue_id}")
            print(f"           → In a real client, you'd now send a follow-up")
            print(f"             request with a task_continue_response to resume.")
            # For this demo we just stop here — resuming requires building
            # the continuation payload which is client-framework-specific.
            break
        elif isinstance(evt, ToolCallDone):
            print(f"[{elapsed}] 🔧 {evt.name}")
        elif isinstance(evt, ContentDelta):
            print(f"[{elapsed}] 💬 {evt.text!r}")

# Uncomment to run
# stream_long_task(ENDPOINT_NAME, LONG_QUERY)

# COMMAND ----------
# MAGIC %md
# MAGIC ## What to take from this notebook
# MAGIC
# MAGIC **For the V1 / V2 streaming conversation:**
# MAGIC
# MAGIC 1. **V1 streaming is essentially free** — the Supervisor endpoint already
# MAGIC    emits well-structured events. Chainlit just needs to consume them and
# MAGIC    map each event type to a UI update (tool badge, token append, done).
# MAGIC
# MAGIC 2. **V2 streaming depends on what you observe in RAW_MODE.** If sub-agent
# MAGIC    inner tokens arrive between tool_call and tool_result → already working.
# MAGIC    If you see one fat `function_call_output` block → V2 needs a custom
# MAGIC    Supervisor or waiting for native forwarding.
# MAGIC
# MAGIC 3. **`task_continue_request` is the long-task safety net.** Multi-source
# MAGIC    queries (4-5 sub-agent calls) will likely exceed the default timeout —
# MAGIC    plan for this in the Chainlit client, not later.
# MAGIC
# MAGIC **For the Apps migration:**
# MAGIC
# MAGIC Port `parse_event()` + the streaming loop structure above into a Chainlit
# MAGIC handler. Replace the `print()` calls with Chainlit UI primitives
# MAGIC (`cl.Message.stream_token()` for text, `cl.Step` for tool calls). That's
# MAGIC the V1 shipping path.
# MAGIC
# MAGIC ## Graduation criteria (when this notebook becomes a numbered quickstart)
# MAGIC
# MAGIC This experiment graduates when all of the following are stable:
# MAGIC - Event shapes verified across multiple Supervisor endpoints (not just one)
# MAGIC - RAW_MODE observations documented (args streaming yes/no, sub-agent forwarding yes/no)
# MAGIC - Chainlit handler pattern validated end-to-end (notebook → Apps deployment)
# MAGIC - `task_continue_request` resume flow implemented (not just detected)
# MAGIC
# MAGIC At that point, retire this script and promote to `10_chainlit_streaming.py`
# MAGIC (or similar) in the main pedagogical flow.
