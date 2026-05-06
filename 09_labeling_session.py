# Databricks notebook source
# MAGIC %md
# MAGIC # 09 — Labeling Sessions, Review App, and Golden-Dataset Round-Trip
# MAGIC
# MAGIC Collect SME feedback on agent traces using MLflow 3 GenAI's labeling session API,
# MAGIC and accumulate that feedback as a versioned UC golden dataset for re-evaluation.
# MAGIC
# MAGIC **What this notebook covers (canonical happy path only):**
# MAGIC 1. Generate traces from your agent (mock here; swap in your real agent for prod)
# MAGIC 2. Build a labeling session with mixed schemas (feedback + expectation)
# MAGIC 3. Hand the URL to SMEs; collect ratings + ground-truth responses
# MAGIC 4. `sync(to_dataset=...)` → labels become a versioned UC golden dataset
# MAGIC 5. Use that dataset in `mlflow.genai.evaluate()` for ongoing regression detection
# MAGIC
# MAGIC **Mental model:**
# MAGIC ```
# MAGIC ┌──────────┐    ┌────────────────┐    ┌────────────────┐    ┌──────────────┐
# MAGIC │  Agent   │ ──►│ Experiment     │ ──►│ Labeling       │ ──►│ Golden       │
# MAGIC │ (traced) │    │ traces         │    │ Session        │    │ Dataset (UC) │
# MAGIC │          │    │                │    │ + schemas:     │    │ assessments  │
# MAGIC │          │    │                │    │   feedback     │    │ + expects    │
# MAGIC │          │    │                │    │   expectation  │    │              │
# MAGIC └──────────┘    └────────────────┘    └────────────────┘    └──────┬───────┘
# MAGIC                                                                    │
# MAGIC                                          mlflow.genai.evaluate ◄───┘
# MAGIC                                          (Correctness, custom scorers, ...)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ⚠️ Caveats — read before you write code
# MAGIC
# MAGIC | # | Caveat | Implication |
# MAGIC |---|---|---|
# MAGIC | 1 | **Use a non-UC-linked experiment** for labeling sessions | Labeling API can't resolve traces stored in UC Delta tables (404). Two-experiment split: UC for monitoring, regular for labeling. |
# MAGIC | 2 | **Schema `type` controls destination, not the name** | `"feedback"` → trace assessments only. `"expectation"` → dataset `expectations` columns on `sync()`. |
# MAGIC | 3 | **Use predefined expectation schema names** (`schemas.EXPECTED_RESPONSE`, `EXPECTED_FACTS`, `GUIDELINES`) | Built-in scorers (`Correctness`, `ExpectationGuidelines`) recognize these by name. Custom names need custom scorers. |
# MAGIC | 4 | **Don't rename `request`/`response` columns** before `merge_records(search_traces_df)` | Renaming strips trace-source detection; records become orphaned. Pass the DF as-is. |
# MAGIC | 5 | **`add_dataset()` requires 100% trace-backed records** | One orphan plain-dict record fails the whole call. No per-record skipping. |
# MAGIC | 6 | **UC datasets persist across runs** | Phantom expectations from prior `merge_records` calls aren't cleared automatically. Drop & recreate to reset. |
# MAGIC | 7 | **Once a schema is bound to any session, it's immutable** | `overwrite=True` fails with "Cannot rename or remove labeling schemas referenced by existing sessions". Version schema names (`*_v1`, `_v2`) for production. |
# MAGIC | 8 | **`add_agent` lives on `ReviewApp`, not `LabelingSession`** | Register agent on review_app first; the `agent=` kwarg on `create_labeling_session` then binds by name. |
# MAGIC | 9 | **Live-agent labeling is deprecated** | The Review App emits "Labeling live agent outputs is no longer officially supported". Replacement: run the agent yourself, capture traces, `add_traces()`. |
# MAGIC | 10 | **Two `get_review_app` exist** | Use `mlflow.genai.labeling.get_review_app` (MLflow 3 native), not `databricks.agents.review_app.get_review_app` (legacy). |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## How to attach work to a session — `add_traces` vs `add_dataset`
# MAGIC
# MAGIC Both attach work to a labeling session, but they have different lifecycle properties.
# MAGIC **Pick `add_dataset` for production; `add_traces` for one-off / dev iteration.**
# MAGIC
# MAGIC | Aspect | `session.add_traces(df_or_list)` | `session.add_dataset(uc_dataset_name)` |
# MAGIC |---|---|---|
# MAGIC | What gets attached | Specific trace IDs (snapshot) | A **named UC dataset** by reference |
# MAGIC | Persistence | None — traces are pinned to this session only | Dataset is durable; can back **many sessions over time** |
# MAGIC | Versioning / lineage | None | UC versioning, lineage, ACLs, audit |
# MAGIC | Fits into eval pipeline | No (manual conversion) | Yes — `mlflow.genai.evaluate(data=dataset)` consumes it directly |
# MAGIC | Re-creatable from prod | You'd re-run `search_traces` each time | Scheduled job appends to the dataset; sessions just point at it |
# MAGIC | Spark / databricks-connect needed | No | Yes (for `create_dataset`/`merge_records`) |
# MAGIC | Polluted-dataset risk | None | Real (caveat 6 — UC datasets persist across runs) |
# MAGIC | Trace-linkage requirement | n/a (records ARE traces) | **All records must be trace-backed** (caveat 5) |
# MAGIC | Best for | Quick dev review, MemAlign feedback collection, ad-hoc batches | Production pipelines, recurring SME review, regression eval |
# MAGIC
# MAGIC **The notebook below shows BOTH:** Steps 1-7 use `add_traces` because it's simpler to
# MAGIC walk through. The "Two-dataset round-trip" pattern at the end shows the production
# MAGIC `add_dataset` flow — that's the one DACHSER should adopt for the recurring loop.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1 — Setup

# COMMAND ----------
# MAGIC %pip install -U "mlflow[databricks]>=3.10" databricks-sdk python-dotenv
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
import os
import time

try:
    dbutils  # noqa: F821
    IS_DATABRICKS = True
except NameError:
    IS_DATABRICKS = False
    from dotenv import load_dotenv
    load_dotenv(override=True)
    os.environ.setdefault("MLFLOW_TRACKING_URI", "databricks")

import mlflow
from databricks.sdk import WorkspaceClient

mlflow.set_tracking_uri("databricks")
w = WorkspaceClient()
CURRENT_USER = w.current_user.me().user_name

# Caveat 1 — use a NON-UC-linked experiment for labeling
EXPERIMENT_PATH = f"/Users/{CURRENT_USER}/labeling-quickstart"
exp = mlflow.get_experiment_by_name(EXPERIMENT_PATH)
EXPERIMENT_ID = exp.experiment_id if exp else mlflow.create_experiment(EXPERIMENT_PATH)
mlflow.set_experiment(EXPERIMENT_PATH)

# UC namespace for the golden dataset
UC_CATALOG = "sandbox"           # ← change to your catalog
UC_SCHEMA = "learning"           # ← change to your schema
GOLDEN_DATASET = f"{UC_CATALOG}.{UC_SCHEMA}.agent_golden_v1"

print(f"User:        {CURRENT_USER}")
print(f"Experiment:  {EXPERIMENT_PATH} (id={EXPERIMENT_ID})")
print(f"Golden ds:   {GOLDEN_DATASET}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2 — Generate traces from your agent
# MAGIC
# MAGIC We use a tiny mock agent here so the notebook is self-contained. To run against your
# MAGIC real agent, replace `mock_agent` with an import of your `ResponsesAgent` (e.g.
# MAGIC `from your_module import AGENT; AGENT.predict(...)`) — `@mlflow.trace` will capture
# MAGIC the same way.

# COMMAND ----------
@mlflow.trace(span_type="TOOL", name="bong_tool")
def bong_tool(echo: str) -> str:
    return f"BONG! (echo={echo})"


@mlflow.trace(span_type="AGENT", name="mock_agent")
def mock_agent(query: str) -> dict:
    q = query.lower()
    if "bong" in q:
        return {"response": f"You said bong. {bong_tool(query)}", "tool_called": True}
    if "ping" in q:
        return {"response": "pong", "tool_called": False}
    return {"response": f"Unknown query: {query!r}", "tool_called": False}


# Generate a small set of traces (mix tool / no-tool / unknown so labels have spread)
queries = [
    "ping",
    "ping ping",
    "bong",
    "say bong twice",
    "what's the weather?",
    "ping me with a bong",
]
for q in queries:
    mock_agent(q)

time.sleep(2)  # let the trace backend index
print(f"✓ Generated {len(queries)} traces")

# COMMAND ----------
# Pull and tag the new traces
traces_df = mlflow.search_traces(
    locations=[EXPERIMENT_ID],
    max_results=50,
    order_by=["timestamp DESC"],
).head(len(queries))

BATCH_TAG = f"batch-{int(time.time())}"
for tid in traces_df["trace_id"]:
    mlflow.set_trace_tag(trace_id=tid, key="batch", value=BATCH_TAG)

time.sleep(2)
print(f"✓ Tagged {len(traces_df)} traces with batch={BATCH_TAG}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3 — Define schemas (mixed: feedback + expectation)
# MAGIC
# MAGIC The pattern here is the cornerstone of this notebook:
# MAGIC - **Feedback schema** (custom, Likert 1-5 + comment) → trace assessments → MemAlign-compatible
# MAGIC - **Expectation schema** (predefined `EXPECTED_RESPONSE`) → dataset `expectations` column → `Correctness`-compatible
# MAGIC
# MAGIC One labeling pass yields BOTH. Sync gives you a golden dataset that built-in scorers
# MAGIC can consume directly.
# MAGIC
# MAGIC **Production tip (caveat 7):** version your custom schema names (`*_v1`, `_v2`).
# MAGIC Once a schema is bound to a session, you can't replace it.

# COMMAND ----------
from mlflow.genai import label_schemas as schemas

CUSTOM_SCHEMA_NAME = "agent_quality_v1"  # version this for production iteration

# Feedback schema — captures subjective rating + free-text rationale (gold for MemAlign)
schemas.create_label_schema(
    name=CUSTOM_SCHEMA_NAME,
    type="feedback",
    title="Response quality (1-5)",
    input=schemas.InputNumeric(min_value=1.0, max_value=5.0),
    instruction=(
        "Rate the response 1 (wrong/hallucinated) to 5 (perfect: right tool, "
        "clear answer, helpful context). Add a comment explaining WHY."
    ),
    enable_comment=True,
    overwrite=True,  # only safe during dev; production: bump the version suffix instead
)
print(f"✓ Feedback schema: {CUSTOM_SCHEMA_NAME}")
# Note: schemas.EXPECTED_RESPONSE is a predefined constant — type="expectation" is baked in

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4 — Create the labeling session (using `add_traces` — simplest path)
# MAGIC
# MAGIC The session combines both schemas. SMEs label both in one pass.
# MAGIC
# MAGIC We use `add_traces` here because it's the most direct way to demo the mechanic.
# MAGIC **For production, prefer the `add_dataset` two-dataset pattern** at the end of this
# MAGIC notebook — see the comparison table above to understand why.

# COMMAND ----------
from mlflow.genai import create_labeling_session

session = create_labeling_session(
    name=f"agent-review-{int(time.time())}",
    label_schemas=[
        CUSTOM_SCHEMA_NAME,            # type="feedback" → trace assessments
        schemas.EXPECTED_RESPONSE,     # type="expectation" → dataset.expectations.expected_response
        # schemas.EXPECTED_FACTS,      # uncomment for fact-list expectations (Correctness scorer)
        # schemas.GUIDELINES,          # uncomment for per-row guidelines (ExpectationGuidelines scorer)
    ],
    assigned_users=[CURRENT_USER],
)
session = session.add_traces(traces_df)

print("=" * 60)
print("LABELING SESSION READY")
print("=" * 60)
print(f"  name:           {session.name}")
print(f"  mlflow_run_id:  {session.mlflow_run_id}")
print(f"  url:            {session.url}")
print("=" * 60)
print("\nShare the URL with your SMEs. For each trace they:")
print("  - Pick a 1-5 rating + leave a comment (feedback schema)")
print("  - Write the ideal response text (expectation schema)")

# COMMAND ----------
# MAGIC %md
# MAGIC ## ⏸ Pause — go label traces in the UI
# MAGIC
# MAGIC Open the URL above. For at least one trace, fill BOTH:
# MAGIC - The 1-5 numeric rating + comment
# MAGIC - The expected_response free-text field
# MAGIC
# MAGIC Then come back and run the next cell.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 5 — Read labels back (verification)

# COMMAND ----------
labeled_df = mlflow.search_traces(run_id=session.mlflow_run_id)
print(f"Traces in session: {len(labeled_df)}")

if "assessments" in labeled_df.columns:
    for _, row in labeled_df.iterrows():
        a = row.get("assessments") or []
        if a:
            print(f"\nTrace {row['trace_id'][:12]}... → {len(a)} assessment(s)")
            for x in a:
                print(f"   {x}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 6 — Sync labels into the golden UC dataset
# MAGIC
# MAGIC `sync(to_dataset=...)` writes both feedback assessments AND expectations into the
# MAGIC dataset. The dataset becomes a versioned ground-truth artifact you re-evaluate
# MAGIC against forever.

# COMMAND ----------
import mlflow.genai.datasets

try:
    golden = mlflow.genai.datasets.create_dataset(name=GOLDEN_DATASET)
    print(f"✓ Created {GOLDEN_DATASET}")
except Exception:
    golden = mlflow.genai.datasets.get_dataset(name=GOLDEN_DATASET)
    print(f"  Reusing {GOLDEN_DATASET}")

session.sync(to_dataset=GOLDEN_DATASET)
print(f"✓ Synced labels from '{session.name}' into {GOLDEN_DATASET}")

# Verify
golden = mlflow.genai.datasets.get_dataset(name=GOLDEN_DATASET)
df = golden.to_df()
print(f"\nDataset rows: {len(df)}")
if "expectations" in df.columns:
    populated = df["expectations"].dropna().apply(lambda x: bool(x))
    print(f"Rows with expectations: {populated.sum()} / {len(df)}")
    print("\nSample expectations:")
    for v in df["expectations"].head(3).tolist():
        print(f"  {v}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 7 — Use the golden dataset in evaluation
# MAGIC
# MAGIC The expectation labels SMEs wrote are now consumable by built-in scorers
# MAGIC like `Correctness` (via `expected_response`) without any additional wiring.

# COMMAND ----------
from mlflow.genai.scorers import Correctness

# predict_fn unpacks `inputs` as kwargs (see GOTCHAS — caveat #X in the eval skill)
def predict_fn(query: str = None, **kwargs) -> dict:
    return mock_agent(query) if query else {"response": ""}

# results = mlflow.genai.evaluate(
#     data=golden,                          # pass the dataset directly
#     predict_fn=predict_fn,
#     scorers=[Correctness()],              # consumes expectations.expected_response
# )
# print(f"Run: {results.run_id}")
# print(results.metrics)

# Uncomment once the dataset has at least one labeled row.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Pattern — Two-dataset round-trip (recommended production setup)
# MAGIC
# MAGIC **This is the pattern to adopt for any recurring SME-review loop** (DACHSER, ongoing
# MAGIC quality monitoring, golden-dataset growth).
# MAGIC
# MAGIC `add_dataset` shines here because the candidates dataset is a **stable UC artifact**
# MAGIC that:
# MAGIC - persists across many labeling sessions
# MAGIC - can be appended to by a scheduled prod-sampling job (see next pattern)
# MAGIC - has UC lineage / ACLs / versioning
# MAGIC - is re-pointable: a new session this week and another next week both reference the
# MAGIC   same dataset name — you just adjust which records go in via `record_ids=`
# MAGIC
# MAGIC Separate the **input** (candidates) from the **output** (curated golden):
# MAGIC ```
# MAGIC candidates_dataset (trace-backed records, append-only)
# MAGIC          │
# MAGIC          │ session.add_dataset(candidates_dataset)
# MAGIC          ▼
# MAGIC    labeling session  ── SMEs label
# MAGIC          │
# MAGIC          │ session.sync(to_dataset=golden_dataset)  ◄── DIFFERENT dataset
# MAGIC          ▼
# MAGIC    golden_dataset (curated, used for evaluate())
# MAGIC ```
# MAGIC
# MAGIC Why this matters:
# MAGIC - Candidates dataset stays raw (and is safe to repopulate from prod sampling)
# MAGIC - Golden dataset is the single source of truth for `evaluate()`
# MAGIC - Avoids the "polluted dataset" problem (caveat 6)
# MAGIC
# MAGIC The code below is a sketch — uncomment + adapt when you're ready to wire it.

# COMMAND ----------
# CANDIDATES_DATASET = f"{UC_CATALOG}.{UC_SCHEMA}.agent_candidates_v1"
# try:
#     candidates = mlflow.genai.datasets.create_dataset(name=CANDIDATES_DATASET)
# except Exception:
#     candidates = mlflow.genai.datasets.get_dataset(name=CANDIDATES_DATASET)
#
# # IMPORTANT (caveat 4): pass search_traces DF AS-IS, no rename.
# # IMPORTANT (caveat 5): only trace-backed rows here, never plain dicts.
# candidates.merge_records(traces_df)
#
# session_v2 = create_labeling_session(
#     name=f"prod-review-{int(time.time())}",
#     label_schemas=[CUSTOM_SCHEMA_NAME, schemas.EXPECTED_RESPONSE],
#     assigned_users=[CURRENT_USER],
# )
# session_v2 = session_v2.add_dataset(dataset_name=CANDIDATES_DATASET)
# # ... SMEs label ...
# session_v2.sync(to_dataset=GOLDEN_DATASET)  # different dataset → curated golden

# COMMAND ----------
# MAGIC %md
# MAGIC ## Pattern — Sampling production traces into a labeling pipeline
# MAGIC
# MAGIC **There is NO native auto-sampling from prod traces into a labeling session or
# MAGIC dataset** (verified May 2026). MLflow gives you two related-but-separate features:
# MAGIC
# MAGIC | Feature | What it does | What it does NOT do |
# MAGIC |---|---|---|
# MAGIC | `MLFLOW_TRACE_SAMPLING_RATIO` / `sampling_ratio_override` | Drops traces at logging time (cost control) | Doesn't route to a dataset |
# MAGIC | `Scorer.start(sampling_config=...)` (Automatic Evaluation) | Runs LLM judges on % of prod traces; assessments land on the trace | Doesn't route to a dataset/session |
# MAGIC
# MAGIC **Recommended pattern: scheduled Databricks Job.**
# MAGIC Daily/weekly, runs the function below. For best signal, prefer sampling **low-scoring**
# MAGIC traces (from auto-eval) over random sampling — those are where SME time pays off.

# COMMAND ----------
def sample_prod_traces_to_candidates(
    *,
    experiment_id: str,
    candidates_dataset_name: str,
    since_hours: int = 24,
    sample_frac: float = 0.1,
    quality_filter: str | None = None,  # e.g., "assessments.correctness < 0.5"
    queue_tag: str = "labeling",
) -> int:
    """
    Sample recent prod traces into a candidates dataset for SME labeling.
    Designed to run as a scheduled Databricks Job.
    """
    import time as _t
    cutoff = int((_t.time() - since_hours * 3600) * 1000)
    # The MLflow filter DSL supports =, IS NULL, IS NOT NULL, AND. Not != or OR.
    # Using IS NULL on the queue tag means "never queued before" — clean dedupe.
    filter_str = (
        f"attributes.timestamp_ms > {cutoff} "
        f"AND attributes.status = 'OK' "
        f"AND tags.queue IS NULL"
    )
    if quality_filter:
        filter_str += f" AND {quality_filter}"

    candidates_traces = mlflow.search_traces(
        locations=[experiment_id],
        filter_string=filter_str,
        max_results=1000,
    )
    if len(candidates_traces) == 0:
        return 0

    sampled = candidates_traces.sample(frac=sample_frac, random_state=42)

    # Tag so they're not re-picked next run
    for tid in sampled["trace_id"]:
        mlflow.set_trace_tag(trace_id=tid, key="queue", value=queue_tag)

    # Caveat 4: pass DF as-is (no rename) so source.trace.trace_id is preserved
    ds = mlflow.genai.datasets.get_dataset(name=candidates_dataset_name)
    ds.merge_records(sampled)
    return len(sampled)

# Example invocation (run inside the scheduled job):
# n = sample_prod_traces_to_candidates(
#     experiment_id=EXPERIMENT_ID,
#     candidates_dataset_name=f"{UC_CATALOG}.{UC_SCHEMA}.agent_candidates_v1",
#     since_hours=24,
#     sample_frac=0.1,
#     quality_filter="assessments.correctness < 0.5",  # only label likely failures
# )
# print(f"Added {n} new candidates to the labeling queue")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Pattern — Vibe-check chat URL (optional)
# MAGIC
# MAGIC Attach a deployed agent endpoint to the Review App so SMEs can ask new questions
# MAGIC interactively. Emits a deprecation warning ("Labeling live agent outputs is no
# MAGIC longer officially supported") — recommended only for ad-hoc exploration, not as
# MAGIC the primary feedback channel.

# COMMAND ----------
# from mlflow.genai import get_review_app
#
# review_app = get_review_app(experiment_id=EXPERIMENT_ID)
# review_app = review_app.add_agent(
#     agent_name="lg_doc_agent",
#     model_serving_endpoint="lg-doc-agent",  # ← your deployed endpoint
#     overwrite=True,
# )
# print(f"Vibe-check chat URL: {review_app.url}/chat")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Cleanup
# MAGIC
# MAGIC ```python
# MAGIC # Drop the golden dataset (irreversible — UC will hold it forever otherwise)
# MAGIC spark.sql(f"DROP TABLE IF EXISTS {GOLDEN_DATASET}")
# MAGIC
# MAGIC # Delete the experiment (also drops associated runs/sessions)
# MAGIC mlflow.delete_experiment(EXPERIMENT_ID)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## What's next
# MAGIC
# MAGIC - **Layer MemAlign** on top of the feedback schema once you have ≥10 labeled traces
# MAGIC   with comments — see `08_evaluation.py` and `learning/phase5-monitoring-eval/05-memalign-judge.py`
# MAGIC - **Schedule the prod sampler** as a Databricks Job so candidates accumulate automatically
# MAGIC - **Wire `mlflow.genai.evaluate(data=GOLDEN_DATASET, scorers=[...])`** into your CI
# MAGIC   for regression detection on every agent change
