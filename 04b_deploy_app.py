# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # 04b — Deploy Agent as Databricks App (alternative to 04)
# MAGIC
# MAGIC **Same agent, different deployment.** This notebook walks through the
# MAGIC `doc-agent-app/` directory — a complete, deployable Databricks App
# MAGIC that packages the same LangGraph agent from notebook 03.
# MAGIC
# MAGIC ## Model Serving (04) vs Apps (this notebook)
# MAGIC
# MAGIC | | Model Serving | Databricks Apps |
# MAGIC |---|---|---|
# MAGIC | **Deploy time** | ~15 min (log → register → provision) | ~2 min (bundle deploy + run) |
# MAGIC | **Iteration** | Re-log + redeploy per change | Redeploy in seconds |
# MAGIC | **Versioning** | Model Registry (artifacts) | Git (code) |
# MAGIC | **Config changes** | Redeploy required | Edit env vars, restart |
# MAGIC | **Agent code** | `class ResponsesAgent` + `predict()` | `@invoke()` / `@stream()` functions |
# MAGIC | **Server** | MLflow Model Server (managed) | MLflow GenAI Server (FastAPI) |
# MAGIC | **Custom routes** | Not possible | Full FastAPI — add any endpoint |
# MAGIC | **Auth** | Service principal only | App auth + user auth (act as user) |
# MAGIC | **Concurrency** | Sync (1 thread = 1 request) | Async capable |
# MAGIC | **Local testing** | Not possible | `uv run start-server` + curl |
# MAGIC | **Cost** | Dedicated compute per endpoint | Shared App compute (2 vCPU, 6GB) |
# MAGIC
# MAGIC **When Model Serving wins:** A/B testing, auto-scaling with GPU, model registry governance.
# MAGIC **When Apps wins:** Fast iteration, custom endpoints, user-level auth, cost-sensitive.

# COMMAND ----------
# MAGIC %md
# MAGIC ## The App Directory
# MAGIC
# MAGIC ```
# MAGIC doc-agent-app/                     ← complete, deployable directory
# MAGIC ├── agent_server/
# MAGIC │   ├── __init__.py
# MAGIC │   └── agent.py                  ← @invoke/@stream agent (the core)
# MAGIC ├── databricks.yml                 ← DAB config: resources + permissions
# MAGIC ├── pyproject.toml                 ← Python dependencies
# MAGIC ├── requirements.txt               ← must contain "uv" (Apps runtime needs it)
# MAGIC └── .env.example                   ← template for local dev config
# MAGIC ```
# MAGIC
# MAGIC Compare with the Model Serving approach:
# MAGIC ```
# MAGIC Model Serving needs:                Apps needs:
# MAGIC   03_agent.py                         doc-agent-app/agent_server/agent.py
# MAGIC   + log_model()                       + databricks.yml
# MAGIC   + register_model()                  (that's it)
# MAGIC   + agents.deploy()
# MAGIC   + pip_requirements=[...]
# MAGIC   + resources=[...]
# MAGIC   + model_config={...}
# MAGIC ```

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1: Understand the Agent Code
# MAGIC
# MAGIC Let's compare the two agent files side by side.

# COMMAND ----------
# Read and display both agent files for comparison
import pathlib

_notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_repo_root = "/Workspace" + "/".join(_notebook_path.split("/")[:-1])

print("=" * 70)
print("  03_agent.py — Model Serving variant (class-based)")
print("=" * 70)
print()
print("  class LangGraphDocAgent(ResponsesAgent):")
print("      def __init__(self):        ← init in class")
print("          self.llm = ...")
print("      def predict(self, req):    ← non-streaming")
print("      def predict_stream(self, req):  ← streaming")
print()
print("  AGENT = LangGraphDocAgent()")
print("  mlflow.models.set_model(AGENT) ← register with MLflow")
print()
print("=" * 70)
print("  doc-agent-app/agent_server/agent.py — Apps variant (functions)")
print("=" * 70)
print()
print("  llm = ChatDatabricks(...)      ← init at module level")
print()
print("  @invoke()")
print("  def non_streaming(req):        ← no class, no self")
print()
print("  @stream()")
print("  def streaming(req):            ← no class, no self")
print()
print("  (no set_model — MLflow discovers @invoke/@stream automatically)")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Key transformation: class → functions
# MAGIC
# MAGIC ```python
# MAGIC # Model Serving — everything inside a class
# MAGIC class LangGraphDocAgent(ResponsesAgent):
# MAGIC     def __init__(self):
# MAGIC         self.llm = ChatDatabricks(endpoint=LLM_ENDPOINT)
# MAGIC         self.llm_with_tools = self.llm.bind_tools(ALL_TOOLS)
# MAGIC
# MAGIC     def predict_stream(self, request):
# MAGIC         graph = self._build_graph()     # self.
# MAGIC         for msg, metadata in graph.stream(...):
# MAGIC             yield ResponsesAgentStreamEvent(
# MAGIC                 item=self.create_function_call_item(...)  # self.helper
# MAGIC             )
# MAGIC
# MAGIC # Apps — flat functions, module-level state
# MAGIC llm = ChatDatabricks(endpoint=LLM_ENDPOINT)         # was self.llm
# MAGIC llm_with_tools = llm.bind_tools(ALL_TOOLS)          # was self.llm_with_tools
# MAGIC
# MAGIC @stream()
# MAGIC def streaming(request):
# MAGIC     graph = _build_graph()              # no self
# MAGIC     for msg, metadata in graph.stream(...):
# MAGIC         yield ResponsesAgentStreamEvent(
# MAGIC             item={...}                  # dict instead of self.helper
# MAGIC         )
# MAGIC ```
# MAGIC
# MAGIC ### Key transformation: config injection
# MAGIC
# MAGIC ```python
# MAGIC # Model Serving — frozen at log_model() time
# MAGIC config = mlflow.models.ModelConfig(development_config={
# MAGIC     "vs_index": "my_catalog.agent_lab.docs_index",
# MAGIC })
# MAGIC VS_INDEX = config.get("vs_index")
# MAGIC
# MAGIC # Apps — env vars, changeable without redeploy
# MAGIC VS_INDEX = os.environ.get("VS_INDEX", "my_catalog.agent_lab.docs_index")
# MAGIC ```

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2: Understand `databricks.yml`
# MAGIC
# MAGIC This is the DAB (Databricks Asset Bundle) config. It declares:
# MAGIC - **What the app is** (name, source code path)
# MAGIC - **What it can access** (resources with permissions)
# MAGIC - **Environment variables** (injected into the app at runtime)

# COMMAND ----------
# Show the databricks.yml
with open(f"{_repo_root}/doc-agent-app/databricks.yml") as f:
    print(f.read())

# COMMAND ----------
# MAGIC %md
# MAGIC ### Resource mapping (Model Serving → Apps)
# MAGIC
# MAGIC In Model Serving, resources are declared in `log_model(resources=[...])`:
# MAGIC ```python
# MAGIC resources = [
# MAGIC     DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT),
# MAGIC     DatabricksServingEndpoint(endpoint_name=EMBEDDING_ENDPOINT),
# MAGIC     DatabricksVectorSearchIndex(index_name=VS_INDEX_NAME),
# MAGIC ]
# MAGIC ```
# MAGIC
# MAGIC In Apps, the same resources go in `databricks.yml`:
# MAGIC ```yaml
# MAGIC resources:
# MAGIC   - name: llm-endpoint
# MAGIC     serving_endpoint:
# MAGIC       name: databricks-gpt-oss-120b
# MAGIC       permission: CAN_QUERY
# MAGIC   - name: vs-index
# MAGIC     uc_securable:
# MAGIC       securable_full_name: my_catalog.agent_lab.docs_index
# MAGIC       securable_type: TABLE
# MAGIC       permission: SELECT
# MAGIC ```
# MAGIC
# MAGIC | MLflow `resources=` | `databricks.yml` resource |
# MAGIC |---|---|
# MAGIC | `DatabricksServingEndpoint` | `serving_endpoint` + CAN_QUERY |
# MAGIC | `DatabricksVectorSearchIndex` | `uc_securable` (TABLE) + SELECT |
# MAGIC | `DatabricksFunction` | `uc_securable` (FUNCTION) + EXECUTE |
# MAGIC | `DatabricksSQLWarehouse` | `sql_warehouse` + CAN_USE |
# MAGIC | `DatabricksGenieSpace` | `genie_space` + CAN_RUN |

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3: Configure for Your Workspace
# MAGIC
# MAGIC Before deploying, edit `doc-agent-app/databricks.yml`:
# MAGIC
# MAGIC 1. Update `securable_full_name` under `vs-index` to match your catalog/schema
# MAGIC 2. Update `env` values to match your actual VS index, LLM endpoint, etc.
# MAGIC 3. (Optional) Change the app `name` if "doc-agent-app" conflicts
# MAGIC
# MAGIC For local testing, copy `.env.example` to `.env` and fill in your values.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4: Deploy
# MAGIC
# MAGIC **From a terminal** (local machine or Databricks Web Terminal):
# MAGIC
# MAGIC ```bash
# MAGIC cd doc-agent-app
# MAGIC
# MAGIC # 1. Validate — catches config errors before deploy
# MAGIC databricks bundle validate --profile <your-profile>
# MAGIC
# MAGIC # 2. Deploy — uploads code, configures resources and permissions
# MAGIC databricks bundle deploy --profile <your-profile>
# MAGIC
# MAGIC # 3. Run — starts/restarts the app (REQUIRED after every deploy!)
# MAGIC databricks bundle run doc_agent --profile <your-profile>
# MAGIC ```
# MAGIC
# MAGIC > **Important:** `bundle deploy` only uploads files. `bundle run` actually
# MAGIC > starts the app. Without `run`, the app keeps serving old code.
# MAGIC
# MAGIC ### Local testing (before deploying)
# MAGIC
# MAGIC ```bash
# MAGIC cd doc-agent-app
# MAGIC cp .env.example .env      # fill in your values
# MAGIC uv sync                   # install dependencies
# MAGIC uv run python start_server.py --reload
# MAGIC ```
# MAGIC
# MAGIC Then test:
# MAGIC ```bash
# MAGIC # Non-streaming
# MAGIC curl -X POST http://localhost:8000/invocations \
# MAGIC   -H "Content-Type: application/json" \
# MAGIC   -d '{"input": [{"role": "user", "content": "What is tool calling?"}]}'
# MAGIC
# MAGIC # Streaming
# MAGIC curl -X POST http://localhost:8000/invocations \
# MAGIC   -H "Content-Type: application/json" \
# MAGIC   -d '{"input": [{"role": "user", "content": "What is tool calling?"}], "stream": true}'
# MAGIC ```

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 5: Test the Deployed App

# COMMAND ----------
# After deploying, test the app endpoint
import requests

# Get these from: databricks apps get doc-agent-app --profile <profile> --output json
APP_URL = ""  # ← paste your app URL here

if APP_URL:
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

    response = requests.post(
        f"{APP_URL.rstrip('/')}/invocations",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={"input": [{"role": "user", "content": "What is tool calling in LangChain?"}]},
    )

    print("=== App Response ===")
    if response.status_code == 200:
        result = response.json()
        # Responses API format — same as Model Serving
        for item in result.get("output", []):
            if item.get("type") == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        print(f"  Agent: {part['text'][:300]}...")
    else:
        print(f"  Error: {response.status_code} {response.text[:200]}")
else:
    print("Deploy the app first (Step 4), then paste APP_URL above.")
    print()
    print("Get it with:")
    print("  databricks apps get doc-agent-app --profile <profile> --output json | jq -r '.url'")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary: Same Agent, Two Paths
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────┐
# MAGIC │                                                                 │
# MAGIC │   03_agent.py ──► 04_deploy_agent.py ──► Serving Endpoint      │
# MAGIC │   (ResponsesAgent class)   (log + register + deploy)           │
# MAGIC │                                         ~15 min                 │
# MAGIC │                                                                 │
# MAGIC │   doc-agent-app/ ──► databricks bundle deploy + run ──► App    │
# MAGIC │   (@invoke/@stream)        (upload + start)                     │
# MAGIC │                            ~2 min                               │
# MAGIC │                                                                 │
# MAGIC │   Both expose: POST /invocations                                │
# MAGIC │   Both accept: {"input": [{"role":"user","content":"..."}]}    │
# MAGIC │   Both return: {"output": [...]}                                │
# MAGIC │                                                                 │
# MAGIC │   Callers (Supervisor, eval, UI) don't know the difference.    │
# MAGIC │                                                                 │
# MAGIC └─────────────────────────────────────────────────────────────────┘
# MAGIC ```
