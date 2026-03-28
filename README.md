# Databricks Agent Quickstart

Deploy a **LangGraph agent** with **Supervisor integration** on Databricks — entirely from workspace notebooks. No CLI, no local tooling, no `brew install` needed.

```
                        SUPERVISOR AGENT
              (routes questions to sub-agents)

  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │ UC Function   │  │ Genie Space  │  │ Knowledge    │
  │ (doc agent)   │  │ (projects)   │  │ Assistant    │
  │               │  │              │  │ (optional)   │
  └──────┬────────┘  └──────┬───────┘  └──────────────┘
         │                  │
         ▼                  ▼
  LangGraph Agent     SQL Warehouse
  (Vector Search)     (Delta table)
```

## What You Get

| Component | Description |
|-----------|-------------|
| **Vector Search index** | LangChain docs with self-computed embeddings |
| **LangGraph agent** | ReAct agent on CPU serving endpoint (scale-to-zero) |
| **UC Function wrapper** | `ai_query()` bridge — makes any endpoint a Supervisor sub-agent |
| **Genie Space** | Natural language SQL on a project tracker table |
| **Supervisor Agent** | Multi-agent orchestrator (KA + custom agent + Genie) |
| **Evaluation notebook** | 10-question comparative eval across agent types |

## Prerequisites

- Databricks workspace with:
  - Unity Catalog enabled
  - Serverless compute available
  - Foundation Model APIs (`databricks-gte-large-en`, `databricks-meta-llama-3-3-70b-instruct`)
- **AgentBricks** enabled for Supervisor (notebooks 01-05 work without it)
- A catalog you have write access to

## Quick Start

### 1. Clone into your workspace

**Workspace sidebar** → **Repos** → **Add Repo** → paste this repo's URL → **Create**

### 2. Configure

Open `_config.py` and set your catalog name:

```python
CATALOG = "my_catalog"    # ← Change to your catalog
SCHEMA = "agent_lab"      # ← Schema will be created automatically
```

### 3. Run notebooks in order

| # | Notebook | What it does | Time |
|---|----------|-------------|------|
| 01 | `01_setup_foundation` | Create catalog + schema + upload sample data | 2 min |
| 02 | `02_create_vs_index` | Compute embeddings + create VS index | 15 min |
| — | `03_agent.py` | Agent runtime file — **don't run directly** | — |
| 04 | `04_deploy_agent` | Log + register + deploy serving endpoint | 15 min |
| 05 | `05_wrap_as_uc_tool` | Wrap endpoint as UC function for Supervisor | 1 min |
| 06 | `06_genie_setup` | Create sample table + Genie Space (UI step) | 2 min |
| 07 | `07_supervisor` | Create Supervisor via REST API | 2 min |
| 08 | `08_evaluation` | Compare Supervisor vs KA vs custom agent | 10 min |
| 99 | `99_cleanup` | **Delete everything** (cost control!) | 2 min |

### 4. Test

Open **AI Playground** → select your Supervisor → try:
- *"What is tool calling in LangChain?"* → routes to doc agent
- *"Which projects are over budget?"* → routes to Genie

### 5. Clean up

**Run `99_cleanup` when done!** VS endpoints bill 24/7.

## Architecture

### Data Pipeline

```
sample_docs.json (bundled)          OR    llms.txt (live download)
        │                                        │
        ▼                                        ▼
  Delta table (docs_chunked)            UC Volume → chunk with Spark
        │
        ▼  ai_query('databricks-gte-large-en', content)
  Delta table (docs_with_embeddings, 1024-dim vectors)
        │
        ▼  Delta Sync (TRIGGERED)
  Vector Search index (self-managed, query_vector required)
```

### Agent Stack

```
  User question
       │
       ▼
  Supervisor Agent (LLM routing)
       │
       ├── "LangChain question?"  ──► UC Function (ask_doc_agent)
       │                                    │
       │                               ai_query() ──► Serving Endpoint
       │                                                    │
       │                                              LangGraph ReAct
       │                                              ├── search_docs (VS)
       │                                              └── LLM response
       │
       ├── "Project data?"  ──► Genie Space
       │                         └── NL → SQL → Delta table
       │
       └── "General knowledge?"  ──► Knowledge Assistant (optional)
```

### The UC Function Trick

Supervisor only supports specific sub-agent types (KA, Genie, UC Function, etc.).
Your custom LangGraph agent is a serving endpoint. The bridge:

```sql
CREATE FUNCTION my_catalog.agent_lab.ask_doc_agent(question STRING)
RETURNS STRING
RETURN SELECT ai_query('langgraph-doc-agent', question)
```

`ai_query()` calls any serving endpoint. Wrap it in a UC function → Supervisor sees it as a tool.

## Cost Guide

| Resource | Billing | How to stop |
|----------|---------|-------------|
| **VS Endpoint** | **24/7** (~$2-5/hr) | Delete endpoint (`99_cleanup`) |
| Serving Endpoint (CPU) | Scale-to-zero ($0 idle) | Delete if not needed |
| Foundation Model APIs | Pay-per-token | Only when used |
| Serverless Compute | Pay-per-second | Auto-stops |
| Delta Tables | Storage only | Minimal |

**Main cost driver:** The VS endpoint. Always delete it when not testing.

## Bundled Data

`data/sample_docs.json` contains 112 pre-chunked LangChain documentation entries
across 10 topics (RAG, agents, tools, memory, streaming, etc.). This allows
instant setup without internet access.

For the full 741-doc corpus, set `DOWNLOAD_LIVE = True` in notebook 01
(downloads from `docs.langchain.com/llms.txt`).

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ai_query()` fails | Serving endpoint may be cold-starting — wait 2 min |
| VS index stuck in PROVISIONING | Wait 10 min. If stuck, delete + recreate |
| `DIRECT_ACCESS not supported` | AgentBricks not enabled — use DELTA_SYNC (default) |
| Supervisor returns 404 | AgentBricks not enabled on this workspace |
| `Unauthorized` on serving | Check resources list in `04_deploy_agent` |
| Import errors after `%pip` | Ensure `dbutils.library.restartPython()` follows `%pip` |

## Customizing

### Different data source

Replace `data/sample_docs.json` with your own chunked documents. Schema:
```json
[{"id": "unique_id", "content": "chunk text...", "source": "filename.md", "chunk_index": 0}]
```

### Different embedding model

Edit `_config.py`:
```python
EMBEDDING_ENDPOINT = "databricks-gte-large-en"  # or "databricks-qwen3-embedding-0-6b" for multilingual
```

Note: changing the embedding model requires recreating the VS index (dimension may differ).

### Adding more tools to the agent

Edit `03_agent.py` — add new `@tool` functions to `ALL_TOOLS`. Then re-run notebooks 04-05 to redeploy.

## License

MIT — see [LICENSE](LICENSE).

## Credits

Extracted from [dbx-agent-lab](https://github.com/errajibadr/dbx-agent-lab), a hands-on
learning course for the Databricks Agent stack (AgentBricks, Mosaic AI, Teams integration).
