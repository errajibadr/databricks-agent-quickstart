# Databricks Agent Quickstart — Project Intelligence

## What This Repo Is
A **standalone, notebook-only path** to deploy a Supervisor multi-agent system on Databricks in ~45 minutes. No CLI, no local tooling, no `brew install`.

A standalone notebook path: clone → configure catalog → run notebooks 01-08 → deployed Supervisor with eval.

## Architecture

```
_config.py  ◄── single source of truth (catalog, schema, endpoints)
    │
    ├── 01_setup_foundation  ── UC hierarchy + sample data (Option A or B)
    ├── 02_create_vs_index   ── embeddings + VS endpoint + Delta Sync index
    ├── 03_agent.py          ── LangGraph ResponsesAgent (NOT run directly)
    ├── 04_deploy_agent      ── log_model → UC register → serving endpoint
    ├── 05_wrap_as_uc_tool   ── ai_query() UC function for Supervisor
    ├── 06_genie_setup       ── project tracker table + Genie Space (UI)
    ├── 07_supervisor        ── Supervisor via REST API (KA, Genie, UC tool, MCP)
    ├── 08_evaluation        ── mlflow.genai.evaluate() with trajectory scorers
    └── 99_cleanup           ── tear down everything (COST CONTROL)
```

See [`experiments/streaming/`](experiments/streaming/) for streaming investigation (which models support token-level streaming with tools bound, and why the Apps variant behaves differently from Model Serving).

## Key Patterns

### Agent querying
```python
from databricks_openai import DatabricksOpenAI
client = DatabricksOpenAI()
response = client.responses.create(model="endpoint-name", input=[...])
```
Both custom agent and Supervisor use the same Responses API pattern.

### Evaluation
Uses `mlflow.genai.evaluate()` (NOT legacy `mlflow.evaluate()`):
- **Custom agent**: `ToolCallCorrectness` + `RetrievalGroundedness` (trace-based)
- **Supervisor**: `make_judge()` routing trajectory judge (response-based)
- Golden dataset: `{"inputs": {...}, "expectations": {"expected_tools": [...], "expected_sub_agent": "..."}, "tags": {...}}`

### Config flow
- `_config.py` = runtime values for notebooks (via `%run ./_config`)
- `03_agent.py` = tries `from _config import ...` for smoke-test, falls back to hardcoded defaults for serving container
- `04_deploy_agent.py` = passes `model_config={...}` at `log_model()` time, overriding `development_config`

## Conventions
- `.py` files use Databricks notebook format (`# Databricks notebook source` / `# COMMAND ----------`)
- All notebooks start with `%run ./_config`
- Notebooks with `%pip install` trigger `restartPython()` then re-run `_config`
- `03_agent.py` is a Python module (imported by 04), NOT a notebook to run directly
- Default LLM: `databricks-gpt-oss-120b`
- Default embedding: `databricks-gte-large-en`

## Cost Awareness
- **VS Endpoint bills 24/7** (~$2-5/hr) — always run `99_cleanup` when done
- Serving endpoint has scale-to-zero — $0 when idle
- Foundation Model APIs are pay-per-token — negligible for this quickstart
- Total cost for a full run: ~$3-8 (mostly VS endpoint time)
