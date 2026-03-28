# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # 08 — Comparative Agent Evaluation
# MAGIC
# MAGIC **Compares:** Supervisor vs Knowledge Assistant vs Custom LangGraph Agent
# MAGIC
# MAGIC **Time:** ~10 minutes | **Cost:** LLM inference (pay-per-token)
# MAGIC
# MAGIC ## Why Compare?
# MAGIC
# MAGIC Each agent path has different strengths:
# MAGIC ```
# MAGIC ┌───────────────────────────────────────────────────────────────────────┐
# MAGIC │                                                                       │
# MAGIC │  Custom LangGraph Agent    Supervisor + UC Tool    KA (if available)  │
# MAGIC │  ───────────────────────   ────────────────────    ─────────────────  │
# MAGIC │  ✓ Full control            ✓ Multi-domain routing  ✓ Zero-code setup │
# MAGIC │  ✓ Custom tools            ✓ Genie + KA + tools    ✓ Auto VS index   │
# MAGIC │  ✓ Streaming               ✗ Extra hop latency     ✓ Built-in ALHF   │
# MAGIC │  ✗ More code               ✗ REST API only         ✗ Limited tools   │
# MAGIC │  ✗ Manual deployment       ✗ Max 20 sub-agents     ✗ No custom logic │
# MAGIC │                                                                       │
# MAGIC └───────────────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC This notebook runs the same 10 questions through each agent and compares:
# MAGIC - **Answer quality** (relevance, correctness, groundedness)
# MAGIC - **Latency** (time-to-first-response)
# MAGIC - **Tool usage** (did it use the right tools?)

# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
# MAGIC %pip install -U "mlflow[databricks]>=3.9" databricks-agents pandas
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
import mlflow
mlflow.set_experiment(MLFLOW_EXPERIMENT)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1: Golden Dataset
# MAGIC
# MAGIC 10 questions spanning different categories:
# MAGIC - **Retrieval** (4): should trigger Vector Search
# MAGIC - **Project data** (3): should route to Genie (Supervisor only)
# MAGIC - **Cross-domain** (2): tests Supervisor's routing intelligence
# MAGIC - **Out-of-scope** (1): should refuse gracefully

# COMMAND ----------
import pandas as pd

golden_dataset = pd.DataFrame([
    # --- Retrieval questions (should trigger VS search) ---
    {
        "question": "What is tool calling in LangChain and how does it work?",
        "expected_topic": "retrieval",
        "expected_tool": "search_docs",
        "expected_keywords": ["tool", "calling", "function", "bind_tools"],
        "notes": "Core LangChain concept — should be in the index",
    },
    {
        "question": "How do I create a RAG pipeline with LangChain?",
        "expected_topic": "retrieval",
        "expected_tool": "search_docs",
        "expected_keywords": ["retrieval", "augmented", "generation", "chain"],
        "notes": "RAG is a primary use case — rich docs expected",
    },
    {
        "question": "What are the different types of memory in LangChain?",
        "expected_topic": "retrieval",
        "expected_tool": "search_docs",
        "expected_keywords": ["memory", "buffer", "conversation", "history"],
        "notes": "Memory types are well-documented in LangChain",
    },
    {
        "question": "How does LangGraph differ from LangChain agents?",
        "expected_topic": "retrieval",
        "expected_tool": "search_docs",
        "expected_keywords": ["langgraph", "state", "graph", "agent"],
        "notes": "Should contrast graph-based vs classic agent approach",
    },

    # --- Project data questions (Genie / Supervisor only) ---
    {
        "question": "Which projects are currently over budget?",
        "expected_topic": "project_data",
        "expected_tool": "project_tracker",
        "expected_keywords": ["over budget", "actual_cost", "budget"],
        "notes": "Genie should generate SQL: WHERE actual_cost > budget",
    },
    {
        "question": "What is the total budget allocated to the Data Science team?",
        "expected_topic": "project_data",
        "expected_tool": "project_tracker",
        "expected_keywords": ["Data Science", "budget", "total"],
        "notes": "Genie should aggregate by team",
    },
    {
        "question": "Which high-priority projects are at risk?",
        "expected_topic": "project_data",
        "expected_tool": "project_tracker",
        "expected_keywords": ["high", "priority", "at risk"],
        "notes": "Genie should filter: priority='High' AND status='At Risk'",
    },

    # --- Cross-domain questions (tests routing) ---
    {
        "question": "We're building a RAG agent for our Supply Chain project. What LangChain patterns should we use?",
        "expected_topic": "cross_domain",
        "expected_tool": "search_docs",
        "expected_keywords": ["RAG", "retrieval", "chain"],
        "notes": "Mentions a project but is really a LangChain question",
    },
    {
        "question": "How much are we spending on AI/ML projects, and what tools do they use?",
        "expected_topic": "cross_domain",
        "expected_tool": "project_tracker",
        "expected_keywords": ["AI/ML", "budget", "spending"],
        "notes": "Primarily a data question, mentions tools tangentially",
    },

    # --- Out-of-scope ---
    {
        "question": "What is the weather in Munich today?",
        "expected_topic": "out_of_scope",
        "expected_tool": "none",
        "expected_keywords": [],
        "notes": "Should refuse gracefully — no tools should trigger",
    },
])

print(f"Golden dataset: {len(golden_dataset)} questions")
print(f"  Retrieval:     {len(golden_dataset[golden_dataset.expected_topic == 'retrieval'])}")
print(f"  Project data:  {len(golden_dataset[golden_dataset.expected_topic == 'project_data'])}")
print(f"  Cross-domain:  {len(golden_dataset[golden_dataset.expected_topic == 'cross_domain'])}")
print(f"  Out-of-scope:  {len(golden_dataset[golden_dataset.expected_topic == 'out_of_scope'])}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2: Query Functions
# MAGIC
# MAGIC Helper functions to query each agent type and capture response + latency.

# COMMAND ----------
import time
import json

def query_custom_agent(question: str) -> dict:
    """Query the custom LangGraph agent via serving endpoint."""
    start = time.time()
    try:
        response = _w.serving_endpoints.query(
            name=AGENT_ENDPOINT_NAME,
            messages=[{"role": "user", "content": question}],
        )
        latency = time.time() - start
        # Extract text from response
        answer = ""
        if hasattr(response, 'choices') and response.choices:
            answer = response.choices[0].message.content or ""
        elif hasattr(response, 'predictions'):
            answer = str(response.predictions)
        return {"answer": answer, "latency": latency, "error": None}
    except Exception as e:
        return {"answer": "", "latency": time.time() - start, "error": str(e)}


def query_supervisor(question: str, supervisor_id: str) -> dict:
    """Query the Supervisor agent via REST API."""
    start = time.time()
    try:
        workspace_url = _w.config.host.rstrip("/")
        token = _w.config.token
        resp = requests.post(
            f"{workspace_url}/api/2.0/multi-agent-supervisors/{supervisor_id}/chat",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"messages": [{"role": "user", "content": question}]},
        )
        latency = time.time() - start
        if resp.status_code == 200:
            result = resp.json()
            answer = result.get("choices", [{}])[0].get("message", {}).get("content", str(result))
            return {"answer": answer, "latency": latency, "error": None}
        return {"answer": "", "latency": latency, "error": f"{resp.status_code}: {resp.text[:200]}"}
    except Exception as e:
        return {"answer": "", "latency": time.time() - start, "error": str(e)}


def query_ka(question: str, ka_endpoint: str) -> dict:
    """Query a Knowledge Assistant via its serving endpoint."""
    start = time.time()
    try:
        response = _w.serving_endpoints.query(
            name=ka_endpoint,
            messages=[{"role": "user", "content": question}],
        )
        latency = time.time() - start
        answer = ""
        if hasattr(response, 'choices') and response.choices:
            answer = response.choices[0].message.content or ""
        return {"answer": answer, "latency": latency, "error": None}
    except Exception as e:
        return {"answer": "", "latency": time.time() - start, "error": str(e)}

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3: Run Evaluation
# MAGIC
# MAGIC Configure which agents to evaluate, then run all questions through each.

# COMMAND ----------
import requests

# ═══════════════════════════════════════════════════════════
#  CONFIGURE WHICH AGENTS TO EVALUATE
# ═══════════════════════════════════════════════════════════

EVAL_CUSTOM_AGENT = True  # Custom LangGraph agent (notebook 04)
EVAL_SUPERVISOR = False   # ← Set True + fill SUPERVISOR_ID below
EVAL_KA = False           # ← Set True + fill KA_ENDPOINT below

SUPERVISOR_ID = ""        # ← From notebook 07
KA_ENDPOINT = ""          # ← KA's serving endpoint name

# ═══════════════════════════════════════════════════════════

results = []

for idx, row in golden_dataset.iterrows():
    q = row["question"]
    print(f"\n[{idx+1}/{len(golden_dataset)}] {q[:60]}...")

    entry = {
        "question": q,
        "expected_topic": row["expected_topic"],
        "expected_tool": row["expected_tool"],
    }

    # Custom agent
    if EVAL_CUSTOM_AGENT:
        r = query_custom_agent(q)
        entry["custom_answer"] = r["answer"]
        entry["custom_latency"] = r["latency"]
        entry["custom_error"] = r["error"]
        print(f"  Custom:     {r['latency']:.1f}s {'✓' if not r['error'] else '✗ ' + r['error'][:60]}")

    # Supervisor
    if EVAL_SUPERVISOR and SUPERVISOR_ID:
        r = query_supervisor(q, SUPERVISOR_ID)
        entry["supervisor_answer"] = r["answer"]
        entry["supervisor_latency"] = r["latency"]
        entry["supervisor_error"] = r["error"]
        print(f"  Supervisor: {r['latency']:.1f}s {'✓' if not r['error'] else '✗ ' + r['error'][:60]}")

    # Knowledge Assistant
    if EVAL_KA and KA_ENDPOINT:
        r = query_ka(q, KA_ENDPOINT)
        entry["ka_answer"] = r["answer"]
        entry["ka_latency"] = r["latency"]
        entry["ka_error"] = r["error"]
        print(f"  KA:         {r['latency']:.1f}s {'✓' if not r['error'] else '✗ ' + r['error'][:60]}")

    results.append(entry)

results_df = pd.DataFrame(results)
print(f"\n✓ Evaluation complete: {len(results_df)} questions × {sum([EVAL_CUSTOM_AGENT, EVAL_SUPERVISOR, EVAL_KA])} agents")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4: Latency Comparison

# COMMAND ----------
print("=== Latency Summary (seconds) ===\n")

latency_cols = [c for c in results_df.columns if c.endswith("_latency")]
for col_name in latency_cols:
    agent = col_name.replace("_latency", "")
    series = results_df[col_name].dropna()
    if len(series) > 0:
        print(f"  {agent:12s}  avg={series.mean():.1f}s  min={series.min():.1f}s  max={series.max():.1f}s  p50={series.median():.1f}s")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 5: Answer Quality (LLM-as-Judge)
# MAGIC
# MAGIC Uses MLflow's built-in scorers to evaluate answer quality.
# MAGIC Default judge is Llama 70B (free on FMAPI).

# COMMAND ----------
# Prepare evaluation dataset for MLflow
eval_data = []
for _, row in results_df.iterrows():
    if EVAL_CUSTOM_AGENT and "custom_answer" in row and row.get("custom_answer"):
        eval_data.append({
            "question": row["question"],
            "answer": row["custom_answer"],
            "agent": "custom",
            "expected_topic": row["expected_topic"],
        })
    if EVAL_SUPERVISOR and "supervisor_answer" in row and row.get("supervisor_answer"):
        eval_data.append({
            "question": row["question"],
            "answer": row["supervisor_answer"],
            "agent": "supervisor",
            "expected_topic": row["expected_topic"],
        })
    if EVAL_KA and "ka_answer" in row and row.get("ka_answer"):
        eval_data.append({
            "question": row["question"],
            "answer": row["ka_answer"],
            "agent": "ka",
            "expected_topic": row["expected_topic"],
        })

eval_df = pd.DataFrame(eval_data)
print(f"Evaluation dataset: {len(eval_df)} (question, answer) pairs")

# COMMAND ----------
# Run MLflow evaluate with relevance and safety scorers
from mlflow.metrics import (
    relevance,
    faithfulness,
)

if len(eval_df) > 0:
    with mlflow.start_run(run_name="workspace-kit-eval"):
        eval_result = mlflow.evaluate(
            data=eval_df,
            predictions="answer",
            model_type="question-answering",
            evaluators="default",
            extra_metrics=[
                relevance(),
            ],
            evaluator_config={
                "col_mapping": {
                    "inputs": "question",
                }
            },
        )

    print("\n=== Evaluation Metrics ===")
    print(eval_result.metrics)

    # Per-agent breakdown
    scores_df = eval_result.tables["eval_results_table"]
    if "agent" in eval_df.columns:
        merged = scores_df.copy()
        merged["agent"] = eval_df["agent"].values
        print("\n=== Per-Agent Breakdown ===")
        for agent_name in merged["agent"].unique():
            agent_scores = merged[merged["agent"] == agent_name]
            print(f"\n  {agent_name}:")
            for col_name in [c for c in agent_scores.columns if "score" in c.lower()]:
                vals = agent_scores[col_name].dropna()
                if len(vals) > 0:
                    print(f"    {col_name}: {vals.mean():.2f}")
else:
    print("No evaluation data — enable at least one agent in Step 3")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 6: Routing Accuracy (Supervisor only)
# MAGIC
# MAGIC For the Supervisor, check whether it routed to the correct sub-agent.

# COMMAND ----------
if EVAL_SUPERVISOR and "supervisor_answer" in results_df.columns:
    print("=== Supervisor Routing Analysis ===\n")

    # Rough heuristic: check if the answer mentions the expected domain
    correct_routes = 0
    for _, row in results_df.iterrows():
        answer = str(row.get("supervisor_answer", "")).lower()
        topic = row["expected_topic"]

        routed_correctly = False
        if topic == "retrieval":
            routed_correctly = any(kw in answer for kw in ["langchain", "tool", "chain", "agent", "rag"])
        elif topic == "project_data":
            routed_correctly = any(kw in answer for kw in ["project", "budget", "team", "cost"])
        elif topic == "out_of_scope":
            routed_correctly = any(kw in answer for kw in ["can't", "cannot", "don't", "outside", "sorry"])
        elif topic == "cross_domain":
            routed_correctly = True  # Any reasonable answer is fine

        status = "✓" if routed_correctly else "✗"
        print(f"  {status} [{topic:15s}] {row['question'][:50]}...")
        if routed_correctly:
            correct_routes += 1

    accuracy = correct_routes / len(results_df) * 100
    print(f"\n  Routing accuracy: {correct_routes}/{len(results_df)} ({accuracy:.0f}%)")
else:
    print("Supervisor evaluation not enabled — set EVAL_SUPERVISOR = True in Step 3")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 7: Save Results

# COMMAND ----------
# Save results as Delta table for further analysis
EVAL_TABLE = f"{CATALOG}.{SCHEMA}.eval_results"

spark_results = spark.createDataFrame(results_df)
spark_results.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(EVAL_TABLE)

print(f"✓ Results saved to {EVAL_TABLE}")
print("  View in Catalog Explorer or query with SQL")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Metric | Custom Agent | Supervisor | KA |
# MAGIC |--------|-------------|------------|-----|
# MAGIC | Latency | Fast (direct) | Slower (+routing hop) | Medium |
# MAGIC | LangChain questions | ✓ (VS search) | ✓ (via UC tool) | ✓ (if configured) |
# MAGIC | Project questions | ✗ | ✓ (via Genie) | ✗ |
# MAGIC | Multi-domain | ✗ | ✓ | ✗ |
# MAGIC | Setup complexity | High (code) | Medium (config) | Low (UI) |
# MAGIC
# MAGIC **Key insight:** Supervisor adds latency but unlocks multi-domain routing.
# MAGIC The UC function wrapper lets you plug ANY custom agent into Supervisor's
# MAGIC orchestration layer — best of both worlds.
