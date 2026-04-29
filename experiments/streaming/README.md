# Streaming on Databricks Serving Endpoints — Investigation Notes

**Date of investigation:** 2026-04-22
**Outcome:** Token-level streaming on Databricks Model Serving **with tools bound** is **model-specific**, not an architectural property of `ResponsesAgent`, Databricks Apps, or client libraries.

This folder is the investigation's self-contained write-up: findings (this README) + the scripts that produced them (all `.py` files beside this README).

---

## TL;DR

For any agent that binds tools (LangGraph, LangChain `create_agent`, OpenAI tool-use pattern) and wants a typewriter UX on Databricks:

| Model | With `stream=true` and `tools=[...]` | Verdict |
|---|---|---|
| `databricks-meta-llama-3-3-70b-instruct` | Emits per-token SSE chunks | ✅ **Use this** |
| `databricks-gpt-oss-120b` | Returns one SSE chunk containing a `[{"type":"reasoning"},{"type":"text"}]` blob at the end | ❌ Collapses |
| `databricks-llama-4-maverick` | Returns one SSE chunk at the end | ❌ Collapses |

This is observable with raw HTTP (no Python client, no LangChain, no LangGraph) — it's a property of the serving endpoint + model combination, not anything downstream.

**Action for this repo:**
- Default LLM for `doc-agent-app` set to `llama-3-3-70b-instruct` (see [`../../doc-agent-app/.env.example`](../../doc-agent-app/.env.example)).
- Notebooks [`../../03_agent.py`](../../03_agent.py) / [`../../04_deploy_agent.py`](../../04_deploy_agent.py) still default to `gpt-oss-120b` for the deployed Model Serving agent — streaming still "works" visually because the ReAct loop emits discrete `function_call` / `function_call_output` / final-text events from the graph; the collapse is only inside the final-text item.

---

## What we tested and in what order

### Round 1 — Raw HTTP streaming (ground truth)

**Script:** [`raw_http_stream.py`](raw_http_stream.py)

What it does: hits `<workspace>/serving-endpoints/<model>/invocations` with a plain `urllib.request` POST, `stream=true`, reads the SSE body line by line. No Databricks SDK, no LangChain. The closest thing to "what does the endpoint actually emit."

**Findings:**
- No-tools case: every tested model streams per-token content chunks as expected.
- With-tools case:
  - `gpt-oss-120b`: one final chunk containing the whole response. No incremental `content` deltas.
  - `llama-4-maverick`: same — one chunk at the end (prior finding, documented at the top of [`exploration_notebook.py`](exploration_notebook.py)).
  - `llama-3-3-70b-instruct`: many content chunks arriving over time — **true token streaming even with tools bound**.

This rules out: client bugs, LangChain bugs, LangGraph bugs, MLflow bugs, Apps container bugs. The endpoint itself is the source of the collapse for gpt-oss-120b / maverick.

### Round 2 — ReAct agent via `create_agent` + `astream`

**Script:** [`react_stream.py`](react_stream.py)

What it does: builds a `create_agent(ChatDatabricks(...), tools=[get_weather])` graph (the standard prebuilt ReAct shape), then consumes `graph.astream(stream_mode=["updates", "messages"])` with millisecond timestamps. This is the async streaming path used by [`../../doc-agent-app/agent_server/agent.py`](../../doc-agent-app/agent_server/agent.py) (Databricks Apps variant).

**Findings (2026-04-22):**

```
MODEL: databricks-meta-llama-3-3-70b-instruct
  [ 1011.0ms] 🔧 TOOL_CALL    get_weather({'city': 'Tokyo'})
  [ 1012.5ms] ✅ TOOL_RESULT  'The weather in Tokyo is sunny and 22°C.'
  [ 1779.2ms] 💬 TEXT_DELTA   'The '
  [ 1787.0ms] 💬 TEXT_DELTA   'weather '
  [ 1795.9ms] 💬 TEXT_DELTA   'in '
  ... (8 total chunks, 15ms mean gap)
  verdict: ✅ STREAMING (token-level)

MODEL: databricks-gpt-oss-120b
  [ 1165.0ms] 🔧 TOOL_CALL    get_weather({'city': 'Tokyo'})
  [ 1167.0ms] ✅ TOOL_RESULT  'The weather in Tokyo is sunny and 22°C.'
  [ 2397.2ms] 💬 TEXT_DELTA   '[{"type": "reasoning", "summar...' ← ONE chunk
  verdict: ❌ SINGLE CHUNK (collapsed)
```

Confirms Round 1's findings propagate cleanly through LangGraph's async streaming layer. Tool calls and tool results stream fine in both cases (because those are structural events in the ReAct graph, not LLM tokens). The final-answer text is where the model-specific collapse shows up.

### Round 3 — End-to-end deployed ResponsesAgent

**Script:** [`../../query_deployed_agent.py`](../../query_deployed_agent.py)

What it does: calls the user's deployed `lg-doc-agent` endpoint (Model Serving) via the Responses API. This is the same prebuilt-ReAct LangGraph agent (see `03_agent.py`), packaged with `predict_stream()` and registered to a Model Serving endpoint.

**Findings:** streams token-by-token cleanly, because the deployed agent uses `llama-3-3-70b-instruct`. If the deployed agent had used `gpt-oss-120b`, we'd expect to see the same "one big text delta" collapse at the client.

---

## What we thought (and were wrong about)

### Wrong Hypothesis 1 — "`ResponsesAgent` is a streaming composer that masks LLM collapse"

**Claim made earlier in the session:** "The ResponsesAgent in Model Serving is a streaming laundromat — it consumes whatever the FM returns (even a single blob) and can still emit fine-grained `function_call` / `function_call_output` / `output_text.delta` events."

**Reality:** Partially right, mostly wrong.
- **Correct:** `function_call` and `function_call_output` events are emitted at LangGraph node boundaries, independent of upstream LLM token streaming. Those ARE structural composition.
- **Wrong:** `output_text.delta` events are a **pass-through**. `predict_stream` yields one delta per `AIMessageChunk.content` — if the LLM gave one chunk, you get one delta. No re-splitting.

In code, `predict_stream` at `02-langgraph-agent.py:428` iterates `graph.stream(stream_mode="messages")` which surfaces raw `AIMessageChunk` objects from `ChatDatabricks.astream()`. Each chunk becomes one SSE event. Nothing magical happens.

### Wrong Hypothesis 2 — "Streaming with tools is universally broken on Databricks FM endpoints"

**Claim made earlier:** "`stream=true` + `tools` on Databricks Model Serving → collapse, regardless of model."

**Reality:** This was an overgeneralization from the `llama-4-maverick` finding in [`exploration_notebook.py`](exploration_notebook.py). `llama-3-3-70b-instruct` streams tokens individually even with tools bound. The correct statement is: "some models collapse, some don't — test each one."

### Wrong Hypothesis 3 — "Maybe it's a network topology issue (in-workspace vs. local)"

**User's hypothesis:** deployed agents stream because they're in-VNet; local clients get buffered differently.

**Reality:** Not the issue. Raw HTTP from a local machine against `llama-3-3-70b-instruct` streams tokens fine. And raw HTTP from the same local machine against `gpt-oss-120b` collapses. Same network, same auth, same client — different models, different behaviors.

---

## What's actually going on (minimal model)

```
Client (OpenAI SDK / urllib / LangChain)
   │  POST /serving-endpoints/<model>/invocations  {stream: true, tools: [...]}
   ▼
Databricks Model Serving front door
   │  passes request to the model's inference container
   ▼
Foundation Model inference
   │
   ├── [Streaming-capable model + tools]
   │     emits SSE chunks with incremental `content` deltas
   │     → client sees typewriter
   │
   └── [Non-streaming model with tools, OR reasoning model]
         emits SSE response, but `content` is buffered until complete
         → client sees one final chunk, possibly with structured blocks
              like [{"type":"reasoning","summary":...}, {"type":"text","text":"..."}]
```

The "collapse" happens **before** the SSE stream starts — the inference pipeline for some models waits to finish its full response (including reasoning pass) before emitting any `content`. The SSE framing is honest: it's actually streaming the one chunk it has, as soon as it's available. There's just only one chunk.

This is likely related to how reasoning / tool-calling models decide their full response before emitting. `gpt-oss-120b` produces a structured `[{"type":"reasoning"}, {"type":"text"}]` payload that requires the whole reasoning pass before the final text is committed.

---

## Practical guidance

### For new agents that need typewriter UX
- **Default to `databricks-meta-llama-3-3-70b-instruct`.**
- Validate with [`react_stream.py`](react_stream.py) whenever you consider switching models — `bind_tools` behavior isn't in any doc, only observable.

### For reasoning models (gpt-oss-120b, future o-series)
- Accept that token streaming is off with tools bound. You still get tool calls, tool results, and the final answer as discrete events — just no typewriter on the final message.
- Handle structured content: `content` may be `[{"type": "reasoning", ...}, {"type": "text", ...}]` even as a single chunk. See [`../../doc-agent-app/agent_server/agent.py`](../../doc-agent-app/agent_server/agent.py) (`_extract_text`) for a working parser.

### For Databricks Apps agent deployments
- The async `graph.astream(stream_mode=["updates", "messages"])` path works fine, as long as the underlying model streams.
- `mlflow.langchain.autolog()` was observed to buffer async streaming — disabled in [`../../doc-agent-app/agent_server/agent.py`](../../doc-agent-app/agent_server/agent.py) with a note. Re-enable once MLflow fixes this.

### For the Supervisor (Agent Bricks)
- Supervisor speaks the Responses API. Whether its final-answer deltas are token-granular depends on its internal model choice (not documented publicly, and may change).
- Open question: does the Supervisor **forward** sub-agent streaming events, or does it buffer tool_result blocks? Investigate with [`../stream_supervisor_demo.py`](../stream_supervisor_demo.py) + `RAW_MODE=True`. Look for `response.output_text.delta` events arriving **between** `function_call` and `function_call_output`.

---

## Reference test scripts in this repo

| Script | Purpose |
|---|---|
| [`raw_http_stream.py`](raw_http_stream.py) | Ground-truth raw HTTP SSE test. Change `LLM_ENDPOINT` at top, run directly. No agent, no tools, just the wire. |
| [`react_stream.py`](react_stream.py) | ReAct (`create_agent`) async streaming test across multiple models. Reports per-token timestamps + verdict. |
| [`streaming_comparison.py`](streaming_comparison.py) | Compares `ChatDatabricks` vs `ChatOpenAI` vs raw HTTP — isolates whether streaming collapse is a client bug or endpoint behavior. |
| [`direct_stream.py`](direct_stream.py) | Layer-isolation tests (raw `ChatDatabricks.astream`, `create_agent` v1/v2 formats) — use to narrow which layer breaks streaming. |
| [`with_tools.py`](with_tools.py) | Quick sanity check: does `create_agent` stream differently with vs. without tools? |
| [`responses_api.py`](responses_api.py) | Smoke test: does `DatabricksOpenAI.responses.create()` work against FM endpoints (no streaming)? |
| [`exploration_notebook.py`](exploration_notebook.py) | Databricks notebook with fuller test matrix (multiple models × tool configs). Superseded by Round 1/2 findings above; kept for model-coverage reference. |
| [`../stream_supervisor_demo.py`](../stream_supervisor_demo.py) | Supervisor Responses API SSE observer. Answers V1/V2 streaming questions for Agent Bricks. |
| [`../../query_deployed_agent.py`](../../query_deployed_agent.py) | End-to-end: calls a deployed ResponsesAgent endpoint via Responses API. Confirms client-side streaming experience. |
| [`../../doc-agent-app/local_server_stream.py`](../../doc-agent-app/local_server_stream.py) | Streams from the local Apps agent server (`localhost:8181`). Apps-specific — requires `start_server.py` running. |

---

## Open questions

1. **Supervisor streaming forwarding** — does `workspace-kit-supervisor` emit sub-agent text deltas between `function_call` and `function_call_output`, or only after? Run [`../stream_supervisor_demo.py`](../stream_supervisor_demo.py) with `RAW_MODE=True`.
2. **Model list coverage** — `claude-sonnet-4`, `gpt-4o`, `gpt-4.1` (if available in the workspace) not yet tested. Add them to [`react_stream.py`](react_stream.py)'s `MODELS` list when accessible.
3. **Reasoning models that do stream** — is there a Databricks FM API flag (e.g. `reasoning_effort`, `stream_options.include_usage`) that unlocks streaming for gpt-oss-style models? Not yet tested.
4. **`predict_stream` re-splitting** — could a custom `ResponsesAgent.predict_stream` artificially chunk a collapsed response into fake deltas to smooth the UX? Technically yes, but it would be cosmetic (the real generation already finished). Not recommended for honesty reasons.
