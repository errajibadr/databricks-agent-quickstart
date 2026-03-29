# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # 08 — Agent Evaluation with MLflow GenAI
# MAGIC
# MAGIC **Evaluates:** Custom LangGraph Agent, Supervisor, KA (comparative)
# MAGIC
# MAGIC **Time:** ~10 minutes | **Cost:** LLM inference (pay-per-token)
# MAGIC
# MAGIC ## What This Does
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────┐
# MAGIC │                                                                 │
# MAGIC │  Golden Dataset (10 questions)                                  │
# MAGIC │       │                                                         │
# MAGIC │       ▼  predict_fn (calls agent via OpenAI client)            │
# MAGIC │                                                                 │
# MAGIC │  mlflow.genai.evaluate()                                       │
# MAGIC │       │                                                         │
# MAGIC │       ├── Guidelines (professional tone, tool usage clarity)   │
# MAGIC │       ├── Correctness (vs expected_facts)                      │
# MAGIC │       ├── RelevanceToQuery                                     │
# MAGIC │       ├── Safety                                               │
# MAGIC │       └── RetrievalGroundedness (traces with RETRIEVER spans)  │
# MAGIC │                                                                 │
# MAGIC │  ──► MLflow Experiment Run (metrics + traces + per-row scores) │
# MAGIC │                                                                 │
# MAGIC └─────────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC **API:** Uses `mlflow.genai.evaluate()` (NOT the legacy `mlflow.evaluate()`).
# MAGIC Data format: `{"inputs": {...}, "expectations": {...}}`.

# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
# MAGIC %pip install -U "mlflow[databricks]>=3.9" databricks-langchain "langgraph>=0.3.4" "lgp>=1.0.0" databricks-agents pydantic pandas
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
import mlflow

mlflow.set_experiment(MLFLOW_EXPERIMENT)
mlflow.langchain.autolog()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1: Golden Dataset
# MAGIC
# MAGIC Uses the `mlflow.genai.evaluate()` format: each record has `inputs` and `expectations`.
# MAGIC
# MAGIC **Categories:**
# MAGIC - Retrieval (4): should trigger Vector Search via `search_docs`
# MAGIC - Project data (3): should route to Genie (Supervisor only)
# MAGIC - Cross-domain (2): tests Supervisor's routing intelligence
# MAGIC - Out-of-scope (1): should refuse gracefully

# COMMAND ----------
eval_data = [
    # ── Retrieval (search_docs tool) ───────────────────────────────────
    {
        "inputs": {"query": "What is tool calling in LangChain and how does it work?"},
        "expectations": {
            "expected_facts": [
                "The response should explain tool/function calling in LangChain",
                "The response should mention bind_tools or tool decorator",
            ],
        },
        "tags": {"category": "retrieval"},
    },
    {
        "inputs": {"query": "How do I create a RAG pipeline with LangChain?"},
        "expectations": {
            "expected_facts": [
                "The response should describe retrieval-augmented generation",
                "The response should mention retriever or vector store",
            ],
        },
        "tags": {"category": "retrieval"},
    },
    {
        "inputs": {"query": "What are the different types of memory in LangChain?"},
        "expectations": {
            "expected_facts": [
                "The response should describe memory types like buffer or conversation history",
            ],
        },
        "tags": {"category": "retrieval"},
    },
    {
        "inputs": {"query": "How does LangGraph differ from LangChain agents?"},
        "expectations": {
            "expected_facts": [
                "The response should contrast graph-based vs classic agent approach",
            ],
        },
        "tags": {"category": "retrieval"},
    },

    # ── Project data (Genie / Supervisor only) ─────────────────────────
    {
        "inputs": {"query": "Which projects are currently over budget?"},
        "expectations": {
            "expected_facts": [
                "The response should reference project budget or cost data",
            ],
        },
        "tags": {"category": "project_data"},
    },
    {
        "inputs": {"query": "What is the total budget allocated to the Data Science team?"},
        "expectations": {
            "expected_facts": [
                "The response should include a budget total for Data Science",
            ],
        },
        "tags": {"category": "project_data"},
    },
    {
        "inputs": {"query": "Which high-priority projects are at risk?"},
        "expectations": {
            "expected_facts": [
                "The response should identify at-risk projects with high priority",
            ],
        },
        "tags": {"category": "project_data"},
    },

    # ── Cross-domain ───────────────────────────────────────────────────
    {
        "inputs": {"query": "We're building a RAG agent for our Supply Chain project. What LangChain patterns should we use?"},
        "expectations": {
            "expected_facts": [
                "The response should focus on LangChain RAG patterns, not project data",
            ],
        },
        "tags": {"category": "cross_domain"},
    },
    {
        "inputs": {"query": "How much are we spending on AI/ML projects, and what tools do they use?"},
        "expectations": {
            "expected_facts": [
                "The response should reference project spending or budget data",
            ],
        },
        "tags": {"category": "cross_domain"},
    },

    # ── Out-of-scope ───────────────────────────────────────────────────
    {
        "inputs": {"query": "What is the weather in Munich today?"},
        "expectations": {
            "expected_response": "The agent should indicate it cannot answer weather questions.",
        },
        "tags": {"category": "out_of_scope"},
    },
]

print(f"Golden dataset: {len(eval_data)} questions")
for cat in ["retrieval", "project_data", "cross_domain", "out_of_scope"]:
    count = sum(1 for d in eval_data if d["tags"]["category"] == cat)
    print(f"  {cat:15s}: {count}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2: Configure Scorers
# MAGIC
# MAGIC Modern MLflow GenAI scorers (replaces legacy `mlflow.metrics.relevance`):
# MAGIC
# MAGIC | Scorer | What it checks | Needs |
# MAGIC |--------|---------------|-------|
# MAGIC | `Guidelines` | Custom rules (tone, clarity) | I/O only |
# MAGIC | `Correctness` | Matches expected_facts | expectations |
# MAGIC | `RelevanceToQuery` | Answer addresses the question | I/O only |
# MAGIC | `Safety` | No harmful content | I/O only |
# MAGIC | `RetrievalGroundedness` | Answer based on retrieved docs | RETRIEVER trace spans |

# COMMAND ----------
from mlflow.genai.scorers import (
    Guidelines,
    Correctness,
    RelevanceToQuery,
    RetrievalGroundedness,
    Safety,
)

scorers = [
    Guidelines(
        name="professional_tone",
        guidelines=[
            "The response must maintain a professional, helpful tone",
            "The response must not include made-up information",
            "If the agent doesn't know, it should say so clearly",
        ],
    ),
    Guidelines(
        name="tool_usage_clarity",
        guidelines=[
            "When the agent uses retrieved documents, it should cite or reference sources",
            "Tool results should be presented clearly, not as raw output",
        ],
    ),
    Correctness(),
    RelevanceToQuery(),
    RetrievalGroundedness(),
    Safety(),
]

print(f"Configured {len(scorers)} scorers")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3: Define Predict Functions
# MAGIC
# MAGIC Each agent type gets its own predict function. The `predict_fn` receives
# MAGIC `**kwargs` from the `inputs` dict — so it gets `query=` as an argument.

# COMMAND ----------
# Get OpenAI client — handles auth automatically in notebooks
client = _w.serving_endpoints.get_open_ai_client()


def predict_custom_agent(query: str) -> str:
    """Query the custom LangGraph agent via Responses API."""
    response = client.responses.create(
        model=AGENT_ENDPOINT_NAME,
        input=[{"role": "user", "content": query}],
    )
    return response.output_text


def predict_supervisor(query: str) -> str:
    """Query the Supervisor agent via REST API."""
    import requests
    workspace_url = _w.config.host.rstrip("/")
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    resp = requests.post(
        f"{workspace_url}/api/2.0/multi-agent-supervisors/{SUPERVISOR_ID}/chat",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"messages": [{"role": "user", "content": query}]},
        timeout=120,
    )
    resp.raise_for_status()
    result = resp.json()
    return result.get("choices", [{}])[0].get("message", {}).get("content", str(result))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4: Run Evaluation
# MAGIC
# MAGIC Each agent gets its own MLflow run — compare them in the Experiment UI.

# COMMAND ----------
# ═══════════════════════════════════════════════════════════
#  CONFIGURE WHICH AGENTS TO EVALUATE
# ═══════════════════════════════════════════════════════════

EVAL_CUSTOM_AGENT = True   # Custom LangGraph agent (notebook 04)
EVAL_SUPERVISOR = False    # ← Set True + fill SUPERVISOR_ID
SUPERVISOR_ID = ""         # ← From notebook 07

# ═══════════════════════════════════════════════════════════

# COMMAND ----------
# Evaluate custom agent
if EVAL_CUSTOM_AGENT:
    print("=== Evaluating: Custom LangGraph Agent ===")
    custom_results = mlflow.genai.evaluate(
        data=eval_data,
        predict_fn=predict_custom_agent,
        scorers=scorers,
    )
    print(f"\nRun ID: {custom_results.run_id}")
    print("\n--- Metrics ---")
    for name, value in sorted(custom_results.metrics.items()):
        print(f"  {name}: {value}")

# COMMAND ----------
# Evaluate Supervisor
if EVAL_SUPERVISOR and SUPERVISOR_ID:
    # Filter to retrieval + out-of-scope for custom agent (it can't do project data)
    # Supervisor gets the full dataset since it routes to Genie for project questions
    print("=== Evaluating: Supervisor Agent ===")
    supervisor_results = mlflow.genai.evaluate(
        data=eval_data,
        predict_fn=predict_supervisor,
        scorers=scorers,
    )
    print(f"\nRun ID: {supervisor_results.run_id}")
    print("\n--- Metrics ---")
    for name, value in sorted(supervisor_results.metrics.items()):
        print(f"  {name}: {value}")
else:
    if not EVAL_SUPERVISOR:
        print("Supervisor evaluation disabled — set EVAL_SUPERVISOR = True")
    elif not SUPERVISOR_ID:
        print("Supervisor evaluation enabled but SUPERVISOR_ID is empty")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 5: Compare in MLflow UI
# MAGIC
# MAGIC 1. Open the experiment in MLflow: Sidebar → Experiments → search for your experiment
# MAGIC 2. Select both runs → **Compare**
# MAGIC 3. View per-row scores in the **Evaluation** tab
# MAGIC 4. Traces are captured automatically — click any row to see the full trace
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │  MLflow Experiment UI                                       │
# MAGIC │                                                             │
# MAGIC │  Run: custom-agent-eval     Run: supervisor-eval           │
# MAGIC │  ├── Metrics                ├── Metrics                    │
# MAGIC │  │   ├── correctness: 0.8   │   ├── correctness: 0.7      │
# MAGIC │  │   ├── relevance: 0.9     │   ├── relevance: 0.85       │
# MAGIC │  │   └── safety: 1.0        │   └── safety: 1.0           │
# MAGIC │  ├── Evaluation tab         ├── Evaluation tab             │
# MAGIC │  │   └── per-row scores     │   └── per-row scores        │
# MAGIC │  └── Traces                 └── Traces                     │
# MAGIC │      └── full agent trace       └── full agent trace       │
# MAGIC └─────────────────────────────────────────────────────────────┘
# MAGIC ```

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 6: Save Golden Dataset to UC (optional)
# MAGIC
# MAGIC Persist the golden dataset as a UC Delta table for reuse across experiments.

# COMMAND ----------
import json

# Save as Delta table
from pyspark.sql.types import StructType, StructField, StringType

eval_rows = [
    {
        "query": d["inputs"]["query"],
        "expected_facts": json.dumps(d["expectations"].get("expected_facts", [])),
        "expected_response": d["expectations"].get("expected_response", ""),
        "category": d["tags"]["category"],
    }
    for d in eval_data
]

schema = StructType([
    StructField("query", StringType(), False),
    StructField("expected_facts", StringType(), True),
    StructField("expected_response", StringType(), True),
    StructField("category", StringType(), False),
])

EVAL_TABLE = f"{CATALOG}.{SCHEMA}.golden_eval_dataset"
df = spark.createDataFrame(eval_rows, schema=schema)
df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(EVAL_TABLE)

print(f"✓ Golden dataset saved to {EVAL_TABLE} ({len(eval_rows)} rows)")
print("  Reuse: SELECT * FROM " + EVAL_TABLE)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | What changed vs legacy `mlflow.evaluate()` | Why it matters |
# MAGIC |---------------------------------------------|----------------|
# MAGIC | `mlflow.genai.evaluate()` | Purpose-built for GenAI agents |
# MAGIC | `inputs` / `expectations` format | Structured ground truth for scorers |
# MAGIC | Built-in scorers (Correctness, Safety, ...) | No custom metric code needed |
# MAGIC | `Guidelines` scorer | Custom rules via natural language |
# MAGIC | Auto-tracing | Every eval row gets a full MLflow trace |
# MAGIC | Per-row scores in UI | Click any row to see why it scored low |
# MAGIC
# MAGIC **Next steps:**
# MAGIC - Add `make_judge()` for domain-specific LLM judges
# MAGIC - Add `ToolCallCorrectness` / `ToolCallEfficiency` scorers
# MAGIC - Run `99_cleanup` when done to stop billing
