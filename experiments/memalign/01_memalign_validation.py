# Databricks notebook source
# MAGIC %md
# MAGIC # MemAlign / Judge Alignment — MLflow 3.13 validation run
# MAGIC
# MAGIC **What we're testing (experiment, not pedagogy):** re-validate the judge-alignment
# MAGIC surface on MLflow ≥ 3.13.0 (released 2026-06-01) before the dedicated DACHSER
# MAGIC session (second week of June 2026). Six checkpoints, V1–V6 below.
# MAGIC
# MAGIC Grounded 2026-06-10 against MLflow 3.13.0 source + mlflow.org/docs.databricks.com.
# MAGIC Prior content (dbx-agent-lab Phase 5 ex 5.5, GOTCHAS) was written on 3.9/3.10 —
# MAGIC two gotchas may be stale, this run decides:
# MAGIC
# MAGIC | # | Checkpoint | Decides |
# MAGIC |---|---|---|
# MAGIC | V1 | dspy + `MemAlignOptimizer` import on 3.13 | dependency story |
# MAGIC | V2 | Databricks-hosted `reflection_lm` | is the April `$defs`/`$ref` proxy rejection fixed? (was: only direct OpenAI/Anthropic worked) |
# MAGIC | V3 | explicit Databricks embedding model | avoids hardcoded `openai:/text-embedding-3-small` default footgun |
# MAGIC | V4 | labeling-session bridge | labels land on trace COPIES under session run; does `align()` see them? |
# MAGIC | V5 | `register()` + `start()` of aligned judge | historically broken (DSPy pickling) — 3.13 claims to fix |
# MAGIC | V6 | `_semantic_memory` non-empty | silent distillation failure gotcha |
# MAGIC
# MAGIC **Run on:** Dbx internal sandbox (needs traces from `03_agent.py` / `08_evaluation.py` runs,
# MAGIC or generates its own minimal ones below).

# COMMAND ----------
# MAGIC %pip install -U "mlflow[databricks]>=3.13.0" dspy databricks-agents
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
import mlflow

print(f"MLflow version: {mlflow.__version__}")  # must be >= 3.13.0
assert tuple(int(x) for x in mlflow.__version__.split(".")[:2]) >= (3, 13), "need >=3.13"

# Non-UC experiment — labeling API can't resolve UC-stored traces (GOTCHA, still assumed true)
EXPERIMENT = "/Shared/memalign-validation"
mlflow.set_tracking_uri("databricks")
exp = mlflow.get_experiment_by_name(EXPERIMENT)
EXPERIMENT_ID = exp.experiment_id if exp else mlflow.create_experiment(EXPERIMENT)
mlflow.set_experiment(EXPERIMENT)

JUDGE_NAME = "memalign_validation_quality"  # label schema name MUST equal this

# COMMAND ----------
# MAGIC %md ## V1 — imports

# COMMAND ----------
from mlflow.genai.judges import make_judge
from mlflow.genai.judges.optimizers import MemAlignOptimizer  # raises if dspy missing

print("V1 OK — MemAlignOptimizer importable")
# 3.13 signature (from source): MemAlignOptimizer(reflection_lm=None, retrieval_k=5,
#                                                 embedding_model=None, embedding_dim=512)
# NOTE: Databricks doc example shows `model=` — source says `reflection_lm=`. Trust source.

# COMMAND ----------
# MAGIC %md ## Setup — base judge + a handful of traces with synthetic HUMAN feedback
# MAGIC
# MAGIC MemAlign has NO hard minimum label count (SIMBA/GEPA enforce ≥10) — 6–8 traces
# MAGIC with deliberately judge-disagreeing human labels are enough to validate mechanics.

# COMMAND ----------
base_judge = make_judge(
    name=JUDGE_NAME,
    instructions=(
        "Rate the quality of the agent's answer to the user's question on a 1-5 scale.\n"
        "User question: {{ inputs }}\nAgent answer: {{ outputs }}\n"
        "5 = precise, grounded, actionable. 1 = wrong or hallucinated."
    ),
    feedback_value_type=float,
    model="databricks",  # managed judge model sentinel — requires databricks-agents
)

# Minimal traced "agent" — good and bad answers on purpose
@mlflow.trace
def toy_agent(question: str, mode: str) -> str:
    answers = {
        "good": "There are 50 documents and 742 chunks in the corpus; 'agents' is the largest topic.",
        "vague": "There are some documents in the knowledge base.",
        "wrong": "The corpus contains exactly 9,999 documents about cooking recipes.",
    }
    return answers[mode]

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
# MAGIC %md ## Attach HUMAN feedback programmatically (the non-Review-App path)
# MAGIC
# MAGIC Naming rule: assessment name == judge name (case-insensitive, stripped) +
# MAGIC `source_type=HUMAN`. On 3.13, traces with NO matching human feedback make
# MAGIC MemAlign raise loudly (used to be silently skipped).

# COMMAND ----------
from mlflow.entities import AssessmentSource, AssessmentSourceType

for trace_id, (q, mode, human_score, rationale) in zip(trace_ids, cases):
    mlflow.log_feedback(
        trace_id=trace_id,
        name=JUDGE_NAME,                      # == judge name — the pairing key
        value=human_score,
        rationale=rationale,                  # MemAlign distills guidelines from these
        source=AssessmentSource(
            source_type=AssessmentSourceType.HUMAN, source_id="badr@validation"
        ),
    )
print("Human feedback logged on all traces")

# COMMAND ----------
# MAGIC %md ## V2 + V3 + V6 — align with Databricks-hosted reflection + embedding

# COMMAND ----------
traces = mlflow.search_traces(locations=[EXPERIMENT_ID], return_type="list")
print(f"Traces for alignment: {len(traces)}")

# V3: explicit Databricks embedding — NEVER rely on the default
#     (hardcoded openai:/text-embedding-3-small → fails without OPENAI_API_KEY).
EMBEDDING = "databricks:/databricks-gte-large-en"

# --- Reflection-model selection (the crux) ---------------------------------
# The reflection_lm distills GUIDELINES via a STRUCTURED (JSON-schema) request.
# Findings (2026-06-16 run):
#   • reflection_lm="databricks"            → Malformed URI (bare sentinel is only
#                                             valid for make_judge model=, NOT here).
#   • reflection_lm="databricks-gpt-oss-120b" → align() returns BUT distillation throws
#                                             INTERNAL_ERROR (upstream) → semantic memory
#                                             EMPTY (episodic-only). gpt-oss can't serve
#                                             the structured distillation call.
# Fix: use a structured-output-capable model. Claude on Databricks stays on-platform
#      (GDPR-safe for DACHSER); OpenAI is last resort (off-platform egress).
# reflection_lm MUST be provider:/model format.
REFLECTION_CANDIDATES = [
    "databricks:/databricks-claude-sonnet-4-5",            # preferred: structured-capable, on-platform
    "databricks:/databricks-meta-llama-3-3-70b-instruct",  # Databricks fallback
    # "openai:/gpt-4o-mini",  # last resort — needs OPENAI_API_KEY + sends data off-platform
]

aligned, optimizer, REFLECTION_MODEL = None, None, None
for rlm in REFLECTION_CANDIDATES:
    opt = MemAlignOptimizer(reflection_lm=rlm, retrieval_k=5, embedding_model=EMBEDDING)
    try:
        cand = base_judge.align(traces, opt)
    except Exception as e:
        print(f"  ✗ {rlm}: align() raised {type(e).__name__}: {e}")
        continue
    # align() returning is NOT success — distillation fails SILENTLY → episodic-only.
    sem = getattr(cand, "_semantic_memory", None)
    if sem:
        print(f"  ✓ {rlm}: {len(sem)} guidelines distilled (full semantic + episodic)")
        aligned, optimizer, REFLECTION_MODEL = cand, opt, rlm
        break
    print(f"  ⚠ {rlm}: align() ran but semantic memory EMPTY (distillation failed) — trying next")
    if aligned is None:           # keep the first episodic-only result as a fallback
        aligned, optimizer, REFLECTION_MODEL = cand, opt, rlm

assert aligned is not None, "all reflection candidates failed even episodic-only alignment"
print(f"\nUsing reflection_lm={REFLECTION_MODEL}  (semantic={'YES' if getattr(aligned,'_semantic_memory',None) else 'NO — episodic only'})")

# COMMAND ----------
# V6 — semantic memory non-empty? (distillation can fail silently and leave episodic-only)
sem = getattr(aligned, "_semantic_memory", None)
if sem:
    print(f"V6 OK — {len(sem)} distilled guidelines:")
    for g in sem:
        print("  •", str(getattr(g, "guideline_text", g))[:120])
else:
    print("V6 ⚠ — semantic memory empty; check driver logs for distillation errors "
          "(episodic memory still provides alignment value)")

# COMMAND ----------
# MAGIC %md ## Quick before/after sanity — does the aligned judge move toward the SME?

# COMMAND ----------
eval_data = [{"inputs": {"question": q, "mode": m}} for q, m, _, _ in cases]

def predict_fn(question: str, mode: str) -> str:
    return toy_agent(question, mode)

for label, judge in [("BASE", base_judge), ("ALIGNED", aligned)]:
    res = mlflow.genai.evaluate(data=eval_data, predict_fn=predict_fn, scorers=[judge])
    # GOTCHA: per-trace scores live in result_df ('state' column, not 'status');
    # metrics dict is empty without aggregations= on make_judge
    col = f"{JUDGE_NAME}/value"
    scores = res.result_df[col].tolist() if col in res.result_df else "see result_df cols"
    print(f"{label}: {scores}  (human: {[c[2] for c in cases]})")

# COMMAND ----------
# MAGIC %md ## V4 — labeling-session bridge (the Review-App path DACHSER actually uses)
# MAGIC
# MAGIC Traces are COPIED into the session; SME labels land on the copies under the
# MAGIC session's MLflow run. Validation: label one trace in the Review App UI, then
# MAGIC confirm `align()` input fetched via the session run carries the feedback.

# COMMAND ----------
from mlflow.genai import label_schemas
from mlflow.genai.labeling import create_labeling_session

label_schemas.create_label_schema(
    name=JUDGE_NAME,  # == judge name, again
    type="feedback",
    title="Answer quality (1-5)",
    input=label_schemas.InputNumeric(min_value=1.0, max_value=5.0),
    instruction="Rate 1-5. Add a comment explaining WHY — comments feed alignment.",
    enable_comment=True,
    overwrite=True,
)
session = create_labeling_session(name="memalign-validation-v1", label_schemas=[JUDGE_NAME])
session.add_traces(mlflow.search_traces(locations=[EXPERIMENT_ID]))
print(f"Label 1-2 traces in the Review App, then run next cell:\n{session.url}")

# COMMAND ----------
# After labeling in UI:
session_traces = mlflow.search_traces(run_id=session.mlflow_run_id, return_type="list")
labeled = [
    t for t in session_traces
    if any(
        a.name.lower().strip() == JUDGE_NAME
        and a.source and a.source.source_type == AssessmentSourceType.HUMAN
        for a in (t.info.assessments or [])
    )
]
print(f"V4: {len(labeled)}/{len(session_traces)} session traces carry matching HUMAN feedback")
if labeled:
    aligned_v2 = aligned.align(labeled, optimizer)  # incremental on 3.13 — fingerprint-skips known traces
    print("V4 OK — incremental re-align from session-run traces succeeded")

# COMMAND ----------
# MAGIC %md ## V5 — register + start for monitoring (the historically broken path)
# MAGIC
# MAGIC 3.13 source engineered around the DSPy thread-lock pickling failure
# MAGIC (`MemoryAugmentedJudge._create_copy` override). Databricks docs still don't
# MAGIC document aligned-judge monitoring — this cell decides what we tell DACHSER.

# COMMAND ----------
from mlflow.genai.scorers import ScorerSamplingConfig

try:
    # register() takes experiment_id=, NOT name= (the name comes from the judge).
    # This registers the MemoryAugmentedJudge itself — the historically DSPy-pickling-broken path.
    registered = aligned.register(experiment_id=EXPERIMENT_ID)
    started = registered.start(sampling_config=ScorerSamplingConfig(sample_rate=0.1))
    print("V5 OK — aligned judge registered + started for monitoring")
    started.stop()  # don't leave a scorer running on the validation experiment
except Exception as e:
    print(f"V5 FAILED: {type(e).__name__}: {e}")
    print(">>> aligned-judge monitoring still not viable — keep 'not yet' messaging for DACHSER")

# COMMAND ----------
# MAGIC %md ## Results → write back
# MAGIC
# MAGIC After running: update (1) `dbx-agent-lab/learning/GOTCHAS.md` (reflection_lm entry,
# MAGIC aligned-monitoring entry), (2) `docs/dachser/engagement/session-13-memalign-plan.md`
# MAGIC (V2/V5 outcomes change the decision points), (3) Phase 5 ex 5.5 (3.13 deltas).
