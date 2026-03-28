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

---

## Before You Start — Workspace Checklist

Run through this checklist **once** before touching any notebook.
If any item fails, fix it first — notebooks will break otherwise.

- [ ] **Unity Catalog enabled** — Sidebar → Catalog → you should see a catalog browser
- [ ] **Serverless compute available** — Create a notebook → try attaching to "Serverless" in the compute dropdown
- [ ] **You have a catalog you can write to** — Sidebar → Catalog → try creating a schema in your catalog. If you can't, ask your admin for `CREATE SCHEMA` + `CREATE TABLE` on a catalog.
- [ ] **Foundation Model APIs available** — Sidebar → Serving → you should see `databricks-gte-large-en` and `databricks-meta-llama-3-3-70b-instruct` in the system endpoints list
- [ ] **AgentBricks enabled** (for notebooks 07+) — Sidebar → look for "Agents" section. If missing, notebooks 01-06 still work (you get everything except Supervisor)
- [ ] **SQL Warehouse exists** — Sidebar → SQL Warehouses → at least one warehouse (any size). Needed for Genie and `ai_query()`

---

## Step-by-Step Setup Guide

### Step 0: Clone the repo into your workspace

1. Sidebar → **Repos**
2. Click **Add Repo**
3. Paste: `https://github.com/errajibadr/databricks-agent-quickstart`
4. Click **Create Repo**
5. You should see all files: `_config.py`, `01_setup_foundation.py`, etc.

> **If Repos doesn't work** (network restrictions): download the repo as ZIP from GitHub, then Workspace → your user folder → Import → upload the ZIP.

### Step 1: Configure (`_config.py`)

1. Open `_config.py` in the workspace editor
2. **Change line 18:** set `CATALOG` to your catalog name
3. That's it — everything else derives from this

```python
# Line 18 — THE ONLY LINE YOU MUST CHANGE
CATALOG = "my_catalog"    # ← Your catalog name here
```

> **How to find your catalog name:** Sidebar → Catalog → the top-level names in the tree are catalogs. Pick one you own or can write to.

### Step 2: Run `01_setup_foundation` (~2 min)

**What it does:** Creates the schema + volume, loads 112 sample doc chunks into a Delta table.

1. Open `01_setup_foundation.py`
2. **Attach to:** Serverless (top-right compute selector)
3. Click **Run All**
4. **Checkpoint:** You should see:
   ```
   ✓ Catalog:  my_catalog
   ✓ Schema:   my_catalog.agent_lab
   ✓ Volume:   /Volumes/my_catalog/agent_lab/documents
   ✓ Saved 112 chunks to my_catalog.agent_lab.docs_chunked
   ```

> **If it fails on `CREATE CATALOG`:** Your user may not have catalog creation rights. Ask admin, or change `CATALOG` in `_config.py` to an existing catalog you can write to.

### Step 3: Run `02_create_vs_index` (~15 min)

**What it does:** Computes embeddings with `ai_query()`, creates a Vector Search endpoint + index.

1. Open `02_create_vs_index.py`
2. Attach to **Serverless**
3. Click **Run All**
4. **This takes time.** The VS endpoint takes 5-10 min to provision. The notebook has polling loops — let it run.
5. **Checkpoint:** You should see:
   ```
   ✓ Embeddings complete: 112 rows
   ✓ Endpoint is ONLINE!
   ✓ Index is ONLINE and synced!
   Query: What is tool calling in LangChain?
     [0.82xx] oss__python__langchain__tools.md
   ```

> **Cost warning:** The VS endpoint bills **24/7** from this point. Run `99_cleanup` when you're done for the day.

> **If embedding fails:** Check that `databricks-gte-large-en` is available in Serving → System endpoints. If it has guardrails enabled, disable them (Serving → endpoint → Edit → Guardrails → Off).

### Step 4: Run `04_deploy_agent` (~15 min)

**What it does:** Logs the LangGraph agent to MLflow, registers in Unity Catalog, deploys to a CPU serving endpoint.

> **Important:** This notebook has a `%pip install` cell that **restarts Python**. The notebook handles this — it re-runs `%run ./_config` after the restart. Just run all cells top to bottom.

1. Open `04_deploy_agent.py`
2. Attach to **Serverless**
3. Click **Run All**
4. **The `%pip install` cell will restart Python** — this is normal. Continue running from the cell after it.
5. The deploy step takes ~15 min. The notebook polls for readiness.
6. **Checkpoint:**
   ```
   ✓ Agent imported successfully
   ✓ Model logged: models:/m-xxxxx
   ✓ Registered: my_catalog.agent_lab.langgraph_doc_agent v1
   ✓ Deployed: langgraph-doc-agent
   ```

> **If `%pip install` cell shows errors:** Run it again. Some transient dependency resolution issues happen on first install.

> **If deployment hangs past 20 min:** Check Serving → your endpoint. If it says "Failed", look at the logs. Most common issue: missing resource in the `resources` list (the notebook pre-fills these).

### Step 5: Run `05_wrap_as_uc_tool` (~1 min)

**What it does:** Creates a SQL function that calls your serving endpoint via `ai_query()`. This is the bridge that lets Supervisor use your custom agent.

1. Open `05_wrap_as_uc_tool.py`
2. Attach to **Serverless**
3. Click **Run All**
4. **Checkpoint:**
   ```
   ✓ Created UC function: my_catalog.agent_lab.ask_doc_agent
   SELECT my_catalog.agent_lab.ask_doc_agent('What is a retrieval chain?')
   → [agent response text]
   ```

> **If `ai_query()` returns empty:** The serving endpoint might be cold (scaled to zero). Wait 2-3 min and re-run the test cell. First request after cold start takes ~60s.

### Step 6: Run `06_genie_setup` + Create Genie Space in UI (~5 min)

**What it does:** Creates a `project_tracker` Delta table (20 rows). Then you create a Genie Space manually in the UI.

1. Open `06_genie_setup.py`
2. Attach to **Serverless**
3. Click **Run All** — this creates the table
4. **Checkpoint:**
   ```
   ✓ Created my_catalog.agent_lab.project_tracker with 20 rows
   ```
5. **Now do the UI step:**
   - Sidebar → **Genie** (or search "New Genie Space" in the search bar)
   - Click **New Genie Space**
   - **Name:** `Project Tracker`
   - **SQL Warehouse:** Select any available warehouse
   - **Tables:** Click **Add tables** → find `my_catalog.agent_lab.project_tracker` → select it
   - **Instructions** (paste this):
     ```
     You help users explore project portfolio data. The table contains
     project tracking data with budgets, timelines, teams, and statuses.
     When users ask about budgets, compare actual_cost vs budget.
     When users ask about risks, filter by status = 'At Risk'.
     ```
   - Click **Save**
   - **Test it:** Type "Which projects are over budget?" — you should get a SQL-generated answer
   - **Copy the Genie Space ID** from the URL bar: `https://<workspace>/genie/rooms/<THIS_IS_THE_ID>`

6. Go back to the notebook → **paste the ID** in the `GENIE_SPACE_ID = ""` cell → run the verification cells

### Step 7: Run `07_supervisor` (~2 min)

**What it does:** Creates a Supervisor Agent via REST API with your sub-agents.

1. Open `07_supervisor.py`
2. **Before running:** Edit the config cell (around line 30):
   - `GENIE_SPACE_ID = "paste-your-id-here"` ← from Step 6
   - `KA_ID = ""` ← leave empty if you don't have a Knowledge Assistant yet
3. Attach to **Serverless**
4. Click **Run All**
5. **Checkpoint:**
   ```
   Configured 2 sub-agents:
     - doc_search_agent (UC_FUNCTION)
     - project_tracker (GENIE)
   ✓ Created Supervisor: <supervisor_id>
   ```

> **If you get 404 or "feature not enabled":** AgentBricks / Supervisor isn't enabled on your workspace. Notebooks 01-06 still work — you just can't create a Supervisor. Test the agent directly via the serving endpoint.

### Step 8: Test in AI Playground

1. Sidebar → **AI Playground**
2. Select your Supervisor agent (or the serving endpoint directly)
3. Test questions:
   - `"What is tool calling in LangChain?"` → should route to doc agent
   - `"Which projects are over budget?"` → should route to Genie
   - `"What's the weather in Munich?"` → should refuse gracefully

### Step 9 (optional): Run `08_evaluation`

Compares response quality and latency across Supervisor vs custom agent vs KA.
Edit the flags at the top to enable/disable each agent type.

### Step 10: CLEAN UP

**Run `99_cleanup` when you're done for the day.**

1. Open `99_cleanup.py`
2. **Change `CONFIRM_DELETE = True`** (line ~15)
3. Run All
4. This deletes: serving endpoint, VS index, VS endpoint (stops 24/7 billing), UC function, tables

> **To resume next time:** Run notebooks 01 → 07 again. The models stay registered in UC — only the compute resources get deleted.

---

## Recovery Guide — When Things Go Wrong

| Situation | What to do |
|-----------|------------|
| **Notebook fails mid-way** | Fix the issue, then **re-run from the failed cell** (not from the top — most cells are idempotent) |
| **VS endpoint stuck in PROVISIONING** | Wait 10 min. If still stuck: delete it (`_w.vector_search_endpoints.delete_endpoint("vs-endpoint-lab")`), then re-run notebook 02 |
| **Serving endpoint shows FAILED** | Check logs in Serving UI → your endpoint → Logs. Usually a missing resource or pip dependency. Fix and redeploy. |
| **`ai_query()` returns empty** | Endpoint is cold-starting (scale-to-zero). Wait 2-3 min, retry. First request takes ~60s. |
| **"Table already exists" errors** | Safe to ignore — notebooks use `IF NOT EXISTS` and `mode("overwrite")` |
| **Lost track of what's running** | Run `99_cleanup` to tear down everything, then start fresh from notebook 01 |
| **Want to iterate on the agent** | Edit `03_agent.py` → re-run only notebooks 04 + 05 (log + deploy + re-wrap) |

---

## File Reference

```
databricks-agent-quickstart/
├── _config.py              ← EDIT THIS FIRST: catalog + schema names
├── 01_setup_foundation.py  ← Catalog/schema/volume + sample data
├── 02_create_vs_index.py   ← Chunk → embed → VS endpoint + index
├── 03_agent.py             ← LangGraph agent (runtime — don't run directly)
├── 04_deploy_agent.py      ← MLflow log → UC register → serving deploy
├── 05_wrap_as_uc_tool.py   ← ai_query() UC function for Supervisor
├── 06_genie_setup.py       ← Project tracker table + Genie instructions
├── 07_supervisor.py        ← Supervisor creation via REST API
├── 08_evaluation.py        ← 10-question comparative evaluation
├── 99_cleanup.py           ← Tear down everything (COST CONTROL!)
└── data/
    ├── sample_docs.json    ← 112 bundled LangChain doc chunks
    └── crawl_docs_simple.py ← Full corpus downloader (optional)
```

## Cost Guide

| Resource | Billing | Estimated cost | How to stop |
|----------|---------|----------------|-------------|
| **VS Endpoint** | **24/7** | ~$2-5/hr | `99_cleanup` or delete in UI |
| Serving Endpoint (CPU) | Scale-to-zero | $0 idle, ~$0.05/hr active | `99_cleanup` |
| Foundation Model APIs | Pay-per-token | ~$0.01 per notebook run | N/A |
| Serverless Compute | Per-second | ~$0.10 per notebook run | Auto-stops |
| Delta Tables | Storage | Negligible | N/A |

**Total cost for a full run:** ~$3-8 (mostly VS endpoint time).
**To minimize:** Run notebooks 01-07 in one sitting, test, then run `99_cleanup` immediately.

## Architecture Deep Dive

### The UC Function Trick (why this matters)

Supervisor only supports specific sub-agent types: KA, Genie, UC Function, Agent, MCP Tool.
A custom LangGraph agent deployed as a serving endpoint doesn't fit any of these natively.

The workaround: `ai_query()` is a built-in SQL function that can call any serving endpoint.
Wrap it in a UC function → Supervisor treats it as a callable tool:

```sql
CREATE FUNCTION my_catalog.agent_lab.ask_doc_agent(question STRING)
RETURNS STRING
RETURN SELECT ai_query('langgraph-doc-agent', question)
```

The function's `COMMENT` becomes the tool description that Supervisor's LLM reads
to decide routing. Write a good comment → better routing accuracy.

### Self-Managed Embeddings (why not managed?)

Managed embeddings are simpler (Databricks computes them during sync) but process
~1 row/second on pay-per-token endpoints. For 112 chunks, that's ~2 min.
For 13K chunks (full corpus), that's ~3.6 hours.

Self-managed: you batch-embed via `ai_query()` in SQL (50 rows/request, parallel).
Same 13K chunks in ~15 min. Tradeoff: you must embed queries yourself at query time.

## Customizing

### Use your own documents

Replace `data/sample_docs.json` with your own data. Required schema:
```json
[{"id": "unique_id", "content": "chunk text...", "source": "filename.md", "chunk_index": 0}]
```

### Change the embedding model

Edit `_config.py` line 28. Note: changing models requires recreating the VS index
(embedding dimensions may differ — GTE is 1024, Qwen3 is configurable up to 1024).

### Add tools to the agent

Edit `03_agent.py`: add `@tool` functions, append to `ALL_TOOLS`. Re-run notebooks 04-05 to redeploy.

---

## License

MIT — see [LICENSE](LICENSE).

## Credits

Built by [dataunboxed.io](https://dataunboxed.io) as part of a hands-on
learning course for the Databricks Agent stack (AgentBricks, Mosaic AI, Teams integration).
