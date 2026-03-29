# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # 04 — Deploy LangGraph Agent
# MAGIC
# MAGIC **Creates:** MLflow model → UC registration → CPU Serving Endpoint
# MAGIC
# MAGIC **Time:** ~15 minutes (mostly waiting for endpoint provisioning)
# MAGIC **Cost:** ~$0.05/hr with scale-to-zero (CPU). $0 when idle.
# MAGIC
# MAGIC ```
# MAGIC 03_agent.py (runtime code)
# MAGIC     │
# MAGIC     ▼  log_model()
# MAGIC MLflow LoggedModel (tracking)
# MAGIC     │
# MAGIC     ▼  register_model()
# MAGIC UC Model (my_catalog.agent_lab.langgraph_doc_agent)
# MAGIC     │
# MAGIC     ▼  agents.deploy()
# MAGIC Serving Endpoint (langgraph-doc-agent, CPU, scale-to-zero)
# MAGIC ```

# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
# MAGIC %pip install -U "mlflow[databricks]>=3.9" databricks-langchain "langgraph>=0.3.4" "lgp>=1.0.0" databricks-agents pydantic
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
# Re-run config after Python restart
# MAGIC %run ./_config

# COMMAND ----------
import mlflow
import pathlib

mlflow.set_experiment(MLFLOW_EXPERIMENT)
mlflow.set_registry_uri("databricks-uc")
print(f"✓ Experiment: {MLFLOW_EXPERIMENT}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1: Import and smoke-test the agent

# COMMAND ----------
import sys

# Add workspace_kit directory to path so we can import 03_agent
_kit_dir = str(pathlib.Path(__file__).parent.resolve()) if "__file__" in dir() else None
if not _kit_dir:
    # In Databricks notebook — use notebook context
    _notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    _kit_dir = "/Workspace" + "/".join(_notebook_path.split("/")[:-1])

if _kit_dir not in sys.path:
    sys.path.insert(0, _kit_dir)

from importlib import import_module
_agent_mod = import_module("03_agent")
AGENT = _agent_mod.AGENT

print("✓ Agent imported successfully")

# COMMAND ----------
from mlflow.types.responses import ResponsesAgentRequest

result = AGENT.predict(ResponsesAgentRequest(
    input=[{"role": "user", "content": "What is tool calling in LangChain?"}]
))

for item in result.output:
    item_type = getattr(item, "type", "")
    if item_type == "message":
        content = getattr(item, "content", [])
        if content:
            text = content[0].text if hasattr(content[0], "text") else str(content[0])
            print(f"✓ Agent responds: {text[:300]}...")
    elif item_type == "function_call":
        print(f"  Tool call: {getattr(item, 'name', '?')}")
    elif item_type == "function_call_output":
        print(f"  Tool result: {str(getattr(item, 'output', ''))[:150]}...")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2: Declare Resources
# MAGIC
# MAGIC Resources = the serving container's auth allowlist. Miss one → 401 at runtime.

# COMMAND ----------
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
)

resources = [
    DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT),
    DatabricksServingEndpoint(endpoint_name=EMBEDDING_ENDPOINT),
    DatabricksVectorSearchIndex(index_name=VS_INDEX_NAME),
]

print(f"✓ Declared {len(resources)} resources")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3: Log Model to MLflow

# COMMAND ----------
# Resolve path to 03_agent.py
_agent_file = str(
    (pathlib.Path(__file__).parent / "03_agent.py").resolve()
    if "__file__" in dir()
    else pathlib.Path(_kit_dir) / "03_agent.py"
)

model_config = {
    "vs_index": VS_INDEX_NAME,
    "llm_endpoint": LLM_ENDPOINT,
    "embedding_endpoint": EMBEDDING_ENDPOINT,
    "system_prompt": (
        "You are a helpful assistant that answers questions about "
        "LangChain documentation using a vector search index. "
        "Always cite your sources when using retrieved documents. "
        "If you don't know the answer, say so honestly."
    ),
}

input_example = {
    "input": [{"role": "user", "content": "What is tool calling in LangChain?"}]
}

with mlflow.start_run():
    model_info = mlflow.pyfunc.log_model(
        name="langgraph-doc-agent",
        python_model=_agent_file,
        resources=resources,
        model_config=model_config,
        pip_requirements=[
            "mlflow[databricks]>=3.9",
            "databricks-langchain",
            "langgraph>=0.3.4",
            "lgp>=1.0.0",
            "databricks-agents",
            "pydantic",
        ],
        input_example=input_example,
    )

print(f"✓ Model logged: {model_info.model_uri}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4: Register in Unity Catalog

# COMMAND ----------
uc_model_info = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=AGENT_MODEL_NAME,
)
print(f"✓ Registered: {uc_model_info.name} v{uc_model_info.version}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 5: Deploy to Serving Endpoint
# MAGIC
# MAGIC Uses `agents.deploy()` — creates a CPU endpoint with scale-to-zero.
# MAGIC Takes ~15 min. The endpoint will scale to zero when idle ($0 cost).
# MAGIC
# MAGIC > **Note:** You may see a warning about "deploying without a feedback model."
# MAGIC > This is safe to ignore. The feedback model was a sidecar endpoint deprecated
# MAGIC > in Dec 2025. MLflow 3 tracing + assessments replaces it — which we already
# MAGIC > have via `ENABLE_MLFLOW_TRACING=true` below.

# COMMAND ----------
from databricks import agents

experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)

deployment = agents.deploy(
    AGENT_MODEL_NAME,
    model_version=uc_model_info.version,
    endpoint_name=AGENT_ENDPOINT_NAME,
    scale_to_zero=True,
    workload_size="Small",
    environment_vars={
        "ENABLE_MLFLOW_TRACING": "true",
        "MLFLOW_EXPERIMENT_ID": experiment.experiment_id,
    },
    tags={"source": "workspace-kit", "phase": "2"},
)
print(f"✓ Deployed: {deployment.endpoint_name}")
print(f"  Query URL: {deployment.query_endpoint}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 6: Test Deployed Endpoint

# COMMAND ----------
# Wait for endpoint to be ready, then test
import time

print("Waiting for endpoint to be ready...")
for i in range(30):
    try:
        ep = _w.serving_endpoints.get(AGENT_ENDPOINT_NAME)
        state = ep.state.ready if ep.state else None
        if state and state.value == "READY":
            print(f"✓ Endpoint is READY after {i * 30}s")
            break
        print(f"  [{i}] State: {state}")
    except Exception:
        print(f"  [{i}] Endpoint not found yet...")
    time.sleep(30)

# COMMAND ----------
# Test with SDK — ResponsesAgent uses the Responses API format (input=), not ChatCompletions (messages=)
import json

response = _w.serving_endpoints.query(
    name=AGENT_ENDPOINT_NAME,
    input=[{"role": "user", "content": "What is a retrieval chain in LangChain?"}],
)

print("=== Endpoint Response ===")
for item in response.output:
    item_type = item.get("type", "")
    if item_type == "message":
        for part in item.get("content", []):
            if part.get("type") == "output_text":
                print(f"  Agent: {part['text'][:300]}...")
    elif item_type == "function_call":
        print(f"  Tool call: {item.get('name', '?')}")
    elif item_type == "function_call_output":
        print(f"  Tool result: {str(item.get('output', ''))[:150]}...")
print(f"\n✓ Endpoint is working! Proceed to notebook 05 (UC tool wrapper).")
