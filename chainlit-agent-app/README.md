# chainlit-agent-app

One Chainlit UI, two backends. Local-import for fast dev loops on the agent code,
deployed Serving Endpoint for production. Both yield Responses-API-native events
so the same UI dispatch handles them with no normalization layer.

> **Status:** Lanes L1 (local agent), L2 (endpoint over the wire), L3 (Apps OBO)
> all working. Additional workspace targets (different host / different
> serving endpoint) are a `databricks.yml` target swap — see "Iteration loops"
> below.

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
    │ LocalAgentBackend    │                │ EndpointBackend          │
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
| L1 | `local` | n/a (in-process) | n/a | Local laptop | ✅ available |
| L2 | `endpoint` | your deployed endpoint | DEFAULT profile / PAT | Local laptop | ✅ available |
| L3 | `endpoint` | your deployed endpoint | OBO header | Databricks App | ✅ available |

---

## Setup — Lane L1 (local-import, fastest dev loop)

```bash
cd databricks-agent-quickstart/chainlit-agent-app

# Install deps. uv recommended; pip works too.
uv sync                 # or: pip install -e .

# Configure environment
cp .env.example .env
# edit .env — set VS_INDEX (required), tweak BACKEND if needed

# Run
chainlit run app.py
# opens on http://localhost:8000 by default
```

### What `.env` must contain for Lane L1

Only one thing is genuinely required:

- **`VS_INDEX`** — the Vector Search index `03_agent.py`'s `search_docs` tool queries.
  No default; `03_agent.py` raises at import if it's missing.

Auth resolves through the standard Databricks chain (same as `WorkspaceClient()`
everywhere else):

| Pattern | When to use | What to set |
|---|---|---|
| **DEFAULT profile** | You've run `databricks configure` once | Nothing. `WorkspaceClient()` finds `~/.databrickscfg` automatically. |
| **Named profile** | DEFAULT points at the wrong workspace | `DATABRICKS_CONFIG_PROFILE=my-workspace-profile` |
| **Explicit PAT** | Headless / CI where no profile exists | `DATABRICKS_HOST=…` + `DATABRICKS_TOKEN=…` |

`LOCAL_AGENT_MODULE` defaults to `../03_agent.py`. To point at a different agent
file, change the env var (path-based loading, so digit-prefixed filenames work).

---

## Setup — Lane L2 (deployed endpoint, local laptop)

Same install as L1, but flip `BACKEND` and add the endpoint name:

```bash
# In .env:
BACKEND=endpoint
ENDPOINT_NAME=doc-agent-quickstart    # or your deployed Serving Endpoint
```

Then `chainlit run app.py` exactly as L1. The same auth chain applies — your
DEFAULT profile / PAT picks up automatically.

L2 is the recommended dev mode once `03_agent.py` is stable: cold-start latency
disappears (the Serving Endpoint stays warm), and the wire-level event stream
matches what the deployed App will see in L3.

---

## Deploy to Databricks Apps — Lane L3 (OBO, production-shaped)

L3 is the path for shipping this UI to a production Databricks App, where each
end user authenticates with their own identity (on-behalf-of / OBO). Per-user
`CAN_QUERY` is enforced at the serving endpoint instead of every user sharing
the App's service principal.

### Prerequisites

You need the Databricks CLI (≥ v0.239 for Apps support) and Terraform installed
locally. Bundle deploys download Terraform on demand and verify it via
HashiCorp's PGP key — that key has expired in the wild more than once, so
installing Terraform yourself is the durable path:

```bash
brew install terraform              # macOS; or download from releases.hashicorp.com
export DATABRICKS_TF_EXEC_PATH=$(which terraform)
export DATABRICKS_TF_VERSION=$(terraform version | head -1 | awk '{print $2}' | sed 's/v//')
```

(Persist by appending those `export`s to `~/.zshrc` / `~/.bashrc`.)

### Two-step deploy

Apps have **two separate lifecycle phases** that bundle deploy doesn't unify:

```
Local working dir
      │ (1) bundle deploy
      ▼
/Workspace/Users/<you>/.bundle/<bundle-name>/<target>/files/   ← bundle artifact path
      │ (2) bundle run <app-key>
      ▼
/Workspace/Users/<App-SP-id>/src/<deployment-id>/   ← immutable App snapshot, what's running
```

So both commands are needed:

```bash
# (1) Upload code to .bundle/, create/update the App resource pointing there
databricks bundle deploy --target dev

# (2) Make a deployment snapshot from .bundle/.../files/ — this is what makes new code live
databricks bundle run chainlit_agent --target dev
```

`chainlit_agent` is the YAML key in `databricks.yml` under `apps:`, NOT the
display name (`chainlit-agent-app`). Same pattern as bundle-targeting jobs/pipelines.

### Iteration loops

| You changed | Run |
|---|---|
| Python code (`app.py`, `auth.py`, `backends/`, `services/`) | `bundle deploy && bundle run chainlit_agent --target dev` |
| Bundle config only (env vars, `user_api_scopes`, resource bindings) | `bundle deploy --target dev` (no `run` needed — Terraform updates the resource without re-snapshotting code) |
| Both at once | `bundle deploy && bundle run chainlit_agent --target dev` |

### Verifying OBO is actually working

After deploy, watch the App logs as you load the URL and send a message:

```bash
databricks apps logs chainlit-agent-app
```

`DBX_AGENT_LOG_EVENTS=1` in `app.yaml` prints per-event timing — you should see
the request stream out with the user's bearer attached, not the App's SP. If
you can flip the App's `user_api_scopes` to remove `serving.serving-endpoints`
and the App suddenly works for *every* user with the same permissions, that
confirms it was running on App-SP auth, not OBO. (Don't actually deploy that
way long-term — but it's a useful one-time A/B if you're not sure.)

---

## Common gotchas (first-deploy survival kit)

Short fixes for the most common first-deploy stumbles. Each row is its own
1-2 line fix; for the dual-auth one in particular, the root-cause analysis
is in the comments of `backends/endpoint.py:from_env`.

| Symptom | Fix |
|---|---|
| `error downloading Terraform: unable to verify checksums signature: openpgp: key expired` | `brew install terraform` + set `DATABRICKS_TF_EXEC_PATH` and `DATABRICKS_TF_VERSION` (see Prerequisites above). |
| `terraform binary at … is X.Y.Z but expected version is 1.5.5. Set DATABRICKS_TF_VERSION to X.Y.Z to continue` | Set `DATABRICKS_TF_VERSION` to whatever `terraform version` reports — the CLI is being conservative; for fresh state, mismatched versions are fine. |
| `Warning: unknown field: env at resources.apps.<name>` | Move `env:` from `databricks.yml` to `app.yaml`. The schema tightened in late 2025 — `app.yaml` is the canonical place for runtime env; `databricks.yml` is for the resource graph only. |
| `validate: more than one authorization method configured: oauth and pat` (after deploy, on first message) | The Apps runtime auto-injects `DATABRICKS_CLIENT_ID/SECRET`; explicit `token=obo_token` makes two auth methods. `WorkspaceClient(host=..., token=obo_token, auth_type="pat")` disambiguates. Already wired into `backends/endpoint.py:from_env`. |
| App resource exists in Apps UI but visiting the URL serves nothing / stale code | You ran `bundle deploy` but not `bundle run chainlit_agent`. Bundle deploy doesn't trigger a deployment snapshot. |
| `'NoneType' object has no attribute 'stream'` from a chat message | `on_chat_start` raised silently and `backend` was never set on the session. Almost always downstream of the auth conflict above. The `on_chat_start` guard in `app.py` should now print the real error as a chat message — check it. |
| 401 / blank screen on first page load | `auth_from_header` returned `None` because `x-forwarded-access-token` was missing. Verify `user_api_scopes: ["serving.serving-endpoints"]` is set in `databricks.yml` and you're reaching the App through the Apps proxy URL (not a direct compute URL). |
| 403 mid-stream when message starts | The OBO token reached the endpoint but the user lacks `CAN_QUERY` on the serving endpoint. Grant the *user* (not just the App's SP) — OBO checks the user's permissions. |

---

## Smoke tests

Once running (any lane), verify the four exit criteria:

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
   wrapper in `03_agent.py` may strip them.

4. **Multi-turn works.** Ask a follow-up that depends on context:
   > "Can you give an example?"

   The agent should answer based on the prior turn's topic.

---

## Layout

```
chainlit-agent-app/
├── app.py                  # Chainlit handlers + inline event.type dispatch
├── auth.py                 # @cl.header_auth_callback (OBO token capture)
├── app.yaml                # Apps runtime spec (entrypoint + env)
├── databricks.yml          # DABs bundle (App resource, scopes, endpoint binding)
├── backends/
│   ├── __init__.py
│   ├── base.py             # Backend Protocol
│   ├── local_agent.py      # LocalAgentBackend (Lane L1)
│   └── endpoint.py         # EndpointBackend (Lanes L2/L3/L4)
├── services/
│   ├── __init__.py
│   ├── event_normalizer.py # Responses-API events → 4-shape dict
│   └── renderer.py         # ChainlitStream (status_msg + text bubble)
├── .chainlit/config.toml   # UI / framework config
├── pyproject.toml          # deps (chainlit, mlflow, databricks-openai, …)
├── .env.example            # template for local dev (committed)
└── .gitignore              # ignores .env, .chainlit caches, etc.
```

---

## Design references

- **Field-name reference for the event dispatch**: `experiments/stream_supervisor_demo.py`
  (sibling directory in this repo)
- **Local agent target**: `../03_agent.py` — `AGENT = LangGraphDocAgent()` at
  module level + `mlflow.models.set_model(AGENT)` is the convention this app's
  `LocalAgentBackend` relies on.
- **Databricks Apps authorization (official docs)**:
  https://docs.databricks.com/aws/en/dev-tools/databricks-apps/auth — covers
  service-principal app auth and on-behalf-of user auth in detail.
- **Bundles + Apps deployment**:
  https://docs.databricks.com/dev-tools/bundles/

## Known gaps

- `task_continue_request` events are silently skipped (v1). Auto-resume will
  land once a concrete envelope shape from a multi-source / long-task query
  surfaces in real usage.
- Additional workspace targets (different host / endpoint) need a separate
  target block in `databricks.yml` — straightforward but not pre-templated.
