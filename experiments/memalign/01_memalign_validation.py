# Databricks notebook source
# MAGIC %md
# MAGIC # Judge Alignment with MemAlign — Walkthrough
# MAGIC
# MAGIC An LLM judge built with `make_judge` scores on *generic* notions of quality. Your
# MAGIC domain experts score on *their* standards. **Judge alignment** bends the judge toward
# MAGIC the expert; **MemAlign** is the optimizer that does it — without fine-tuning anything.
# MAGIC
# MAGIC MemAlign learns from human-labeled traces into **two memories**, then injects them into
# MAGIC the judge's prompt at scoring time:
# MAGIC
# MAGIC - **Semantic memory** — generalized *rules* (guidelines) distilled once from the expert
# MAGIC   rationales. Applies to every score.
# MAGIC - **Episodic memory** — *few-shot examples* retrieved per query. For each new trace it
# MAGIC   finds the most-similar labeled examples and shows the judge how the expert scored them.
# MAGIC
# MAGIC This notebook walks the full loop end to end on a tiny synthetic example so the mechanics
# MAGIC are easy to follow. Requires MLflow ≥ 3.13.0.

# COMMAND ----------
# MAGIC %pip install -U "mlflow[databricks]>=3.13.0" dspy databricks-agents
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %md ## 1. Setup
# MAGIC
# MAGIC Use a **non-UC-linked experiment** for alignment: the labeling API can't resolve traces
# MAGIC stored in UC Delta tables (it 404s). Keep UC experiments for production monitoring and a
# MAGIC regular experiment for alignment/labeling.

# COMMAND ----------
import mlflow

print(f"MLflow version: {mlflow.__version__}")
assert tuple(int(x) for x in mlflow.__version__.split(".")[:2]) >= (3, 13), "need >=3.13"

EXPERIMENT = "/Shared/memalign-demo"
mlflow.set_tracking_uri("databricks")
exp = mlflow.get_experiment_by_name(EXPERIMENT)
EXPERIMENT_ID = exp.experiment_id if exp else mlflow.create_experiment(EXPERIMENT)
mlflow.set_experiment(EXPERIMENT)

# This single string ties three things together (see notes below). It MUST be identical
# everywhere: the judge name, the human-feedback name, and the label-schema name.
JUDGE_NAME = "answer_quality"

from mlflow.genai.judges import make_judge
from mlflow.genai.judges.optimizers import MemAlignOptimizer  # requires dspy

# COMMAND ----------
# MAGIC %md ## 2. Define the base judge
# MAGIC
# MAGIC MemAlign is scorer-agnostic — float, boolean, or categorical all work. Here a 1–5 scale.

# COMMAND ----------
base_judge = make_judge(
    name=JUDGE_NAME,
    instructions=(
        "Rate the quality of the agent's answer to the user's question on a 1-5 scale.\n"
        "User question: {{ inputs }}\nAgent answer: {{ outputs }}\n"
        "5 = precise, grounded, actionable. 1 = wrong or hallucinated."
    ),
    feedback_value_type=float,
    model="databricks",  # managed judge model — requires databricks-agents
)

# COMMAND ----------
# MAGIC %md ## 3. Generate a few traces
# MAGIC
# MAGIC A minimal traced "agent" that returns deliberately good and bad answers, so the expert
# MAGIC labels below have something to disagree with the generic judge about.

# COMMAND ----------
@mlflow.trace
def toy_agent(question: str, mode: str) -> str:
    answers = {
        "good": "There are 50 documents and 742 chunks in the corpus; 'agents' is the largest topic.",
        "vague": "There are some documents in the knowledge base.",
        "wrong": "The corpus contains exactly 9,999 documents about cooking recipes.",
    }
    return answers[mode]

# (question, mode, expert_score, expert_rationale)
cases = [
    ("How many documents are in the corpus?", "good", 5.0, "Precise, correct numbers, adds context."),
    ("How many documents are in the corpus?", "vague", 2.0, "Technically responsive but useless — no numbers."),
    ("How many documents are in the corpus?", "wrong", 1.0, "Hallucinated count and topic."),
    ("What's the biggest topic?", "good", 4.0, "Correct, could cite chunk counts per topic."),
    ("What's the biggest topic?", "vague", 2.0, "Does not answer the question."),
    ("Summarize the corpus.", "wrong", 1.0, "Invented content — unacceptable."),
]

trace_ids = []
for q, mode, _, _ in cases:
    toy_agent(q, mode)
    trace_ids.append(mlflow.get_last_active_trace_id())
print(f"Generated {len(trace_ids)} traces")

# COMMAND ----------
# MAGIC %md ## 4. Attach expert feedback
# MAGIC
# MAGIC In production these labels come from SMEs in the Review App (Section 7). Here we attach
# MAGIC them programmatically to keep the demo self-contained.
# MAGIC
# MAGIC **Naming rule:** the feedback `name` must equal the judge `name` (case-insensitive) and
# MAGIC have `source_type=HUMAN`. That equality is the *only* thing that pairs a human label to
# MAGIC the judge during `align()`.

# COMMAND ----------
from mlflow.entities import AssessmentSource, AssessmentSourceType

for trace_id, (q, mode, expert_score, rationale) in zip(trace_ids, cases):
    mlflow.log_feedback(
        trace_id=trace_id,
        name=JUDGE_NAME,                      # == judge name — the pairing key
        value=expert_score,
        rationale=rationale,                  # MemAlign distills guidelines from these
        source=AssessmentSource(
            source_type=AssessmentSourceType.HUMAN, source_id="demo-sme"
        ),
    )
print("Expert feedback logged on all traces")

# COMMAND ----------
# MAGIC %md ## 5. Align the judge with MemAlign
# MAGIC
# MAGIC Three models are in play — don't conflate them:
# MAGIC
# MAGIC | Model | Role | When used |
# MAGIC |---|---|---|
# MAGIC | judge model (`make_judge(model=)`) | does the scoring | every score |
# MAGIC | `reflection_lm` | distills guidelines from labels | once, during `align()` |
# MAGIC | `embedding_model` | embeds examples for episodic retrieval | `align()` + each score |
# MAGIC
# MAGIC **Reflection-model gotcha (important).** Guideline distillation asks the reflection model
# MAGIC for *structured* (JSON-schema) output. MemAlign's schema nests a `Guideline` model inside a
# MAGIC `Guidelines` wrapper, so the emitted schema contains a JSON-Schema `$ref`. Databricks
# MAGIC Foundation Model APIs **do not support `$ref`** in `response_format` (documented), and
# MAGIC neither does Groq — so a `databricks:/...` reflection model produces **episodic-only**
# MAGIC alignment (no distilled guidelines), and `align()` does NOT raise — it logs the error and
# MAGIC continues silently. OpenAI and **Azure OpenAI** support `$ref`, so they distill guidelines
# MAGIC fully. To keep data in-tenant, call **Azure OpenAI directly** via `azure:/` (not through a
# MAGIC `databricks:/` endpoint, which re-inserts the gateway that blocks `$ref`).
# MAGIC
# MAGIC Always set `embedding_model` explicitly — it defaults to `openai:/text-embedding-3-small`,
# MAGIC which fails without an OpenAI key.

# COMMAND ----------
traces = mlflow.search_traces(locations=[EXPERIMENT_ID], return_type="list")
print(f"Traces for alignment: {len(traces)}")

EMBEDDING = "databricks:/databricks-gte-large-en"

# reflection_lm must be provider:/model. Tried in order; we accept the first that
# actually distills guidelines (semantic memory non-empty), else fall back to episodic-only.
REFLECTION_CANDIDATES = [
    # "azure:/<your-azure-openai-deployment>",  # in-tenant, supports $ref (needs AZURE_API_* env)
    "openai:/gpt-4o-mini",                       # supports $ref → full distillation
    "databricks:/databricks-meta-llama-3-3-70b-instruct",  # episodic-only fallback ($ref blocked)
]

aligned, optimizer, REFLECTION_MODEL = None, None, None
for rlm in REFLECTION_CANDIDATES:
    opt = MemAlignOptimizer(reflection_lm=rlm, retrieval_k=5, embedding_model=EMBEDDING)
    try:
        cand = base_judge.align(traces, opt)
    except Exception as e:
        print(f"  x {rlm}: align() raised {type(e).__name__}: {e}")
        continue
    # align() returning is NOT success — distillation can fail silently → episodic-only.
    sem = getattr(cand, "_semantic_memory", None)
    if sem:
        print(f"  + {rlm}: {len(sem)} guidelines distilled (semantic + episodic)")
        aligned, optimizer, REFLECTION_MODEL = cand, opt, rlm
        break
    print(f"  ! {rlm}: align() ran but no guidelines distilled (episodic-only) — trying next")
    if aligned is None:
        aligned, optimizer, REFLECTION_MODEL = cand, opt, rlm  # keep as fallback

assert aligned is not None, "all reflection candidates failed even episodic-only alignment"
print(f"\nUsing reflection_lm={REFLECTION_MODEL}  "
      f"(semantic={'YES' if getattr(aligned, '_semantic_memory', None) else 'NO — episodic only'})")

# COMMAND ----------
# MAGIC %md ## 6. Inspect the distilled guidelines
# MAGIC
# MAGIC These are the rules MemAlign extracted from the expert rationales — the "semantic memory".

# COMMAND ----------
sem = getattr(aligned, "_semantic_memory", None)
if sem:
    print(f"{len(sem)} distilled guidelines:")
    for g in sem:
        print("  -", str(getattr(g, "guideline_text", g))[:160])
else:
    print("Semantic memory empty — distillation failed (see Section 5 gotcha). "
          "Episodic memory still provides few-shot alignment value.")

# COMMAND ----------
# MAGIC %md ## 7. Before / after — does the aligned judge move toward the expert?

# COMMAND ----------
eval_data = [{"inputs": {"question": q, "mode": m}} for q, m, _, _ in cases]

def predict_fn(question: str, mode: str) -> str:
    return toy_agent(question, mode)

for label, judge in [("BASE   ", base_judge), ("ALIGNED", aligned)]:
    res = mlflow.genai.evaluate(data=eval_data, predict_fn=predict_fn, scorers=[judge])
    # Per-trace scores live in result_df; the 'state' column (not 'status') holds run state.
    col = f"{JUDGE_NAME}/value"
    scores = res.result_df[col].tolist() if col in res.result_df else "see result_df cols"
    print(f"{label}: {scores}  (expert: {[c[2] for c in cases]})")

# COMMAND ----------
# MAGIC %md ## 8. Review App labeling session (how SMEs label in production)
# MAGIC
# MAGIC Traces are copied into the session; SME labels land on the copies under the session's run.
# MAGIC The label-schema `name` must equal the judge name, same as the programmatic path above.

# COMMAND ----------
from mlflow.genai import label_schemas
from mlflow.genai.labeling import create_labeling_session

label_schemas.create_label_schema(
    name=JUDGE_NAME,  # == judge name
    type="feedback",
    title="Answer quality (1-5)",
    input=label_schemas.InputNumeric(min_value=1.0, max_value=5.0),
    instruction="Rate 1-5. Add a comment explaining WHY — comments feed alignment.",
    enable_comment=True,
    overwrite=True,
)
session = create_labeling_session(name="memalign-demo", label_schemas=[JUDGE_NAME])
session.add_traces(mlflow.search_traces(locations=[EXPERIMENT_ID]))
print(f"Label a trace or two in the Review App, then run the next cell:\n{session.url}")

# COMMAND ----------
# Run after labeling in the Review App UI.
session_traces = mlflow.search_traces(run_id=session.mlflow_run_id, return_type="list")
labeled = [
    t for t in session_traces
    if any(
        a.name.lower().strip() == JUDGE_NAME
        and a.source and a.source.source_type == AssessmentSourceType.HUMAN
        for a in (t.info.assessments or [])
    )
]
print(f"{len(labeled)}/{len(session_traces)} session traces carry matching expert feedback")
if labeled:
    # Alignment is incremental — re-aligning fingerprint-skips traces already learned.
    aligned = aligned.align(labeled, optimizer)
    print("Incremental re-align from the labeling session succeeded")

# COMMAND ----------
# MAGIC %md ## 9. Register the aligned judge for monitoring
# MAGIC
# MAGIC Persist the aligned judge to the experiment and (optionally) start it as a production
# MAGIC monitor that scores a sample of live traffic.

# COMMAND ----------
from mlflow.genai.scorers import ScorerSamplingConfig

try:
    # register() takes experiment_id= (the name comes from the judge).
    registered = aligned.register(experiment_id=EXPERIMENT_ID)
    monitor = registered.start(sampling_config=ScorerSamplingConfig(sample_rate=0.1))
    print("Aligned judge registered and started for monitoring (10% sample)")
    monitor.stop()  # stop so it doesn't keep scoring the demo experiment
except Exception as e:
    print(f"Register/monitor not available here: {type(e).__name__}: {e}")
