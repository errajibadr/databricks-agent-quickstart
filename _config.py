# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # Workspace Kit — Configuration
# MAGIC
# MAGIC **Single source of truth** for all resource names across the workspace kit.
# MAGIC Every notebook starts with `%run ./_config` — change values here once.
# MAGIC
# MAGIC ```
# MAGIC _config.py  ◄── YOU EDIT THIS
# MAGIC     │
# MAGIC     ├── 01_setup_foundation.py
# MAGIC     ├── 02_create_vs_index.py
# MAGIC     ├── 03_agent.py (standalone — reads from model_config, not this)
# MAGIC     ├── 04_deploy_agent.py
# MAGIC     ├── 05_wrap_as_uc_tool.py
# MAGIC     ├── 06_genie_setup.py
# MAGIC     ├── 07_supervisor.py
# MAGIC     ├── 08_evaluation.py
# MAGIC     └── 99_cleanup.py
# MAGIC ```

# COMMAND ----------
# ═══════════════════════════════════════════════════════════
#  EDIT THESE VALUES FOR YOUR WORKSPACE
# ═══════════════════════════════════════════════════════════

# Unity Catalog hierarchy
CATALOG = "my_catalog"
SCHEMA = "agent_lab"

# Volume for raw document files
VOLUME_NAME = "documents"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}"

# Delta tables
TABLE_CHUNKS = f"{CATALOG}.{SCHEMA}.docs_chunked"
TABLE_EMBEDDINGS = f"{CATALOG}.{SCHEMA}.docs_with_embeddings"
TABLE_GENIE = f"{CATALOG}.{SCHEMA}.project_tracker"

# Vector Search
VS_ENDPOINT_NAME = "vs-endpoint-lab"
VS_INDEX_NAME = f"{CATALOG}.{SCHEMA}.docs_index"

# Embedding & LLM endpoints (Foundation Model APIs — already available)
EMBEDDING_ENDPOINT = "databricks-gte-large-en"
LLM_ENDPOINT = "databricks-gpt-oss-120b"

# Agent serving endpoint
AGENT_ENDPOINT_NAME = "langgraph-doc-agent"
AGENT_MODEL_NAME = f"{CATALOG}.{SCHEMA}.langgraph_doc_agent"

# UC Function (wrapper for Supervisor integration)
UC_TOOL_FUNCTION = f"{CATALOG}.{SCHEMA}.ask_doc_agent"

# MLflow experiment
MLFLOW_EXPERIMENT = f"/Users/{{user}}/workspace-kit-agent"

# Supervisor
SUPERVISOR_NAME = "workspace-kit-supervisor"

# COMMAND ----------
# ═══════════════════════════════════════════════════════════
#  DERIVED VALUES (don't edit below unless you know why)
# ═══════════════════════════════════════════════════════════

# Get current user for experiment path
from databricks.sdk import WorkspaceClient

_w = WorkspaceClient()
CURRENT_USER = _w.current_user.me().user_name
MLFLOW_EXPERIMENT = f"/Users/{CURRENT_USER}/workspace-kit-agent"

print("╔═══════════════════════════════════════════════════════╗")
print("║         Workspace Kit — Configuration Loaded          ║")
print("╠═══════════════════════════════════════════════════════╣")
print(f"║  Catalog:      {CATALOG:<40s}║")
print(f"║  Schema:       {SCHEMA:<40s}║")
print(f"║  VS Endpoint:  {VS_ENDPOINT_NAME:<40s}║")
print(f"║  VS Index:     {VS_INDEX_NAME:<40s}║")
print(f"║  Agent:        {AGENT_ENDPOINT_NAME:<40s}║")
print(f"║  User:         {CURRENT_USER:<40s}║")
print("╚═══════════════════════════════════════════════════════╝")
