# chainlit-agent-app

One Chainlit UI, two backends. Local-import for fast dev loops on the agent code,
deployed Serving Endpoint for production. Both yield Responses-API-native events
so the same UI dispatch handles them with no normalization layer.

> **Status:** Step A only — Lane L1 (LocalAgentBackend) is the only working path
> right now. Step B layers on EndpointBackend; Step D ships the Databricks Apps
> deploy. See `memory_bank/creative_phases/creative_phase_2026-04-27_dbx_apps_streaming_agents.md`
> in `dbx-agent-lab/` for the full plan.

## Architecture

```
┌─────────────────────────── Chainlit UI ────────────────────────────┐
│  cl.Message.stream_token()  ←  response.output_text.delta          │
│  cl.Step (open/close)        ←  function_call / function_call_output│
│  cl.Step (collapsed)         ←  reasoning items                    │
└──────────────────────────────────────┬─────────────────────────────┘
                                       │
                          @cl.on_message — single dispatch on `event.type`
                                       │
                              ┌────────▼────────┐
                              │ Backend.stream()│
                              └────────┬────────┘
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
    ┌──────────────────────┐                ┌──────────────────────────┐
    │ LocalAgentBackend    │                │ EndpointBackend  (Step B)│
    │  importlib + AGENT   │                │  AsyncDatabricksOpenAI   │
    │  predict_stream()    │                │  client.responses.create │
    │  yields native       │                │  yields native           │
    │  ResponsesAgent      │                │  ResponseStreamEvent     │
    │  StreamEvent         │                │                          │
    └──────────────────────┘                └──────────────────────────┘
```

## Testing lanes

| Lane | `BACKEND` | Endpoint | Auth | Where | Status |
|---|---|---|---|---|---|
| L1 | `local` | n/a (in-process) | n/a | Local laptop | **available now (Step A)** |
| L2 | `endpoint` | `doc-agent-quickstart` | PAT (`.env`) | Local laptop | Step B |
| L3 | `endpoint` | `doc-agent-quickstart` | OBO header | Personal Dbx App | Step D |
| L4 | `endpoint` | DACHSER Supervisor | OBO header | DACHSER Citrix | Step E |

## Setup (Lane L1)

```bash
cd databricks-agent-quickstart/chainlit-agent-app

# Install deps. uv recommended; pip works too.
uv sync                 # or: pip install -e .

# Configure environment
cp .env.example .env
# edit .env — set DATABRICKS_HOST, DATABRICKS_TOKEN, VS_INDEX

# Run
chainlit run app.py
# opens on http://localhost:8000 by default
```

### What `.env` must contain for Lane L1

Only one thing is genuinely required:

- **`VS_INDEX`** — the Vector Search index `03_agent.py`'s `search_docs` tool queries.
  No default; `03_agent.py` raises at import if it's missing.

Auth resolves through the standard Databricks chain (same as `WorkspaceClient()`
everywhere else), so pick whichever pattern fits your setup:

| Pattern | When to use | What to set |
|---|---|---|
| **DEFAULT profile** | You've run `databricks configure` once | Nothing. `WorkspaceClient()` finds `~/.databrickscfg` automatically. |
| **Named profile** | DEFAULT points at the wrong workspace | `DATABRICKS_CONFIG_PROFILE=my-workspace-profile` |
| **Explicit PAT** | Headless / CI / Apps / Citrix where no profile exists | `DATABRICKS_HOST=…` + `DATABRICKS_TOKEN=…` |

The same chain governs Step B's `AsyncDatabricksOpenAI(workspace_client=…)` —
swapping `BACKEND=local` ↔ `endpoint` doesn't change how auth is resolved.

`LOCAL_AGENT_MODULE` defaults to `../03_agent.py`. To point at a different agent
file, change the env var (path-based loading, so digit-prefixed filenames work).

## Smoke tests

Once `chainlit run app.py` boots, verify Step A's exit criteria:

1. **Token streaming works.** Ask a generic question:
   > "Hi! What can you help me with?"
   Tokens should arrive incrementally, not in one block.

2. **Tool Steps render.** Ask something doc-search-shaped:
   > "How does LangChain handle streaming responses?"
   You should see a `search_docs` Step open with arguments, then close with the
   retrieved chunks as output.

3. **Reasoning Steps render** *(gpt-oss-120b only)*. The default LLM endpoint is
   `databricks-gpt-oss-120b`. If it emits `reasoning` items, they should appear
   as collapsed Steps. If you don't see any, that's also fine — the LangGraph
   wrapper in `03_agent.py` may strip them. (Q5 in the design doc.)

4. **Multi-turn works.** Ask a follow-up that depends on context:
   > "Can you give an example?"
   The agent should answer based on the prior turn's topic.

## Layout

```
chainlit-agent-app/
├── app.py                # Chainlit handlers + inline event.type dispatch
├── backends/
│   ├── __init__.py
│   ├── base.py           # Backend Protocol
│   └── local_agent.py    # LocalAgentBackend (Step A)
├── .chainlit/config.toml # UI / framework config
├── pyproject.toml        # deps
├── .env.example          # template (committed)
└── .gitignore            # ignores .env, .chainlit caches, etc.
```

`endpoint.py`, `databricks.yml`, `app.yaml` arrive in Steps B and D.

## Design references

- Full creative phase: `dbx-agent-lab/memory_bank/creative_phases/creative_phase_2026-04-27_dbx_apps_streaming_agents.md`
- Field-name reference for the dispatch: `databricks-agent-quickstart/experiments/stream_supervisor_demo.py`
- Local agent target: `databricks-agent-quickstart/03_agent.py:224` (`AGENT = LangGraphDocAgent()`)

## Known gaps in Step A

- `EndpointBackend` not yet implemented — `BACKEND=endpoint` raises `NotImplementedError`.
- `task_continue_request` events are silently skipped (v1). Auto-resume lands when
  DACHSER multi-source query testing surfaces a concrete envelope shape (Q7).
- No DAB / `app.yaml` — Step D ships the Apps deploy.
