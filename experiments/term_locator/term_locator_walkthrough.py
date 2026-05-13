# Databricks notebook source
# MAGIC %md
# MAGIC # Term Locator — Domain-Keyword Routing for Supervisor Agents
# MAGIC
# MAGIC **Problem:** The Supervisor LLM router decides which sub-agent to call based on tool descriptions
# MAGIC alone. Domain-specific terminology — product names like `Entago`, internal codes, brand jargon —
# MAGIC is opaque to it. Stuffing keyword lists into tool descriptions doesn't scale (hundreds of names ×
# MAGIC N domains saturates the routing prompt).
# MAGIC
# MAGIC **Solution sketched in Session 10 §5 → validated here:** a small **routing-helper** tool that runs
# MAGIC a pure BM25 search over the existing knowledge index and tells the Supervisor *where* the term
# MAGIC appears, NOT the answer. The Supervisor then routes to the matched domain's sub-agent with
# MAGIC confidence.
# MAGIC
# MAGIC **The flow:**
# MAGIC
# MAGIC ```
# MAGIC                            ┌──────────────────────────────┐
# MAGIC                            │ User: "How does Entago       │
# MAGIC                            │  packaging work for FR-DE?"  │
# MAGIC                            └──────────────┬───────────────┘
# MAGIC                                           │
# MAGIC                                           ▼
# MAGIC             ┌─────────────────────────────────────────────────────────┐
# MAGIC             │ AgentBricks Supervisor (LLM router)                     │
# MAGIC             │  · sees "Entago" — not in tool descriptions             │
# MAGIC             │  · extracts technical terms → calls term_locator first  │
# MAGIC             └──────────────────────────┬──────────────────────────────┘
# MAGIC                                        │  terms=["Entago"]
# MAGIC                                        ▼
# MAGIC             ┌─────────────────────────────────────────────────────────┐
# MAGIC             │ term_locator  (UC function / sub-tool)                  │
# MAGIC             │  ┌──────────────────────────────────────────────────┐   │
# MAGIC             │  │ VS index .similarity_search(                     │   │
# MAGIC             │  │   query_type="FULL_TEXT", num_results=50)        │   │
# MAGIC             │  └──────────────────────────┬───────────────────────┘   │
# MAGIC             │                             │ raw hits                  │
# MAGIC             │                             ▼                           │
# MAGIC             │  ┌──────────────────────────────────────────────────┐   │
# MAGIC             │  │ aggregate BM25 score by source (or domain)       │   │
# MAGIC             │  │ rank → top 3, flag ambiguity                     │   │
# MAGIC             │  └──────────────────────────┬───────────────────────┘   │
# MAGIC             └─────────────────────────────┼───────────────────────────┘
# MAGIC                                           │
# MAGIC                                           ▼
# MAGIC             ┌─────────────────────────────────────────────────────────┐
# MAGIC             │ [{name:"myNET",            domain:"EL", score:64.2},    │
# MAGIC             │  {name:"Network Manual EL",domain:"EL", score: 8.2}]    │
# MAGIC             └─────────────────────────────┬───────────────────────────┘
# MAGIC                                           │
# MAGIC                                           ▼
# MAGIC             ┌─────────────────────────────────────────────────────────┐
# MAGIC             │ Supervisor routes with confidence → european_logistics  │
# MAGIC             └─────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC **Why this is NOT the `query_all_sources` tool we removed:** that one fanned out *answer-retrieval*
# MAGIC across all domains (ambiguity problem). This one returns *where to look*, not the answer — the
# MAGIC Supervisor still routes to a single domain sub-agent for the actual response.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Setup
# MAGIC
# MAGIC Two ways to authenticate:
# MAGIC - **Inside Dbx workspace:** `VectorSearchClient()` with no args — uses the notebook user's identity.
# MAGIC - **Outside / from an app:** service principal credentials (what we use below — matches DACHSER's app deployment pattern).
# MAGIC
# MAGIC The SP needs `USE_CATALOG` + `USE_SCHEMA` + `SELECT` on the source table, and `READ` on the VS endpoint.

# COMMAND ----------

import os
from collections import defaultdict
from typing import Optional

from databricks.vector_search.client import VectorSearchClient

# Inside a Dbx notebook, you can skip env vars and let the SDK pick up the notebook identity.
# Outside (or from a Databricks App), provide SP credentials.
WORKSPACE_URL    = os.environ.get("WORKSPACE_URL")
SP_CLIENT_ID     = os.environ.get("SP_CLIENT_ID")
SP_CLIENT_SECRET = os.environ.get("SP_CLIENT_SECRET")

ENDPOINT_NAME = "vector_search_endpoint"
INDEX_NAME    = "rag_np.default.consolidated_knowledge_silver_vs_index"

if SP_CLIENT_ID and SP_CLIENT_SECRET:
    vsc = VectorSearchClient(
        workspace_url=WORKSPACE_URL,
        service_principal_client_id=SP_CLIENT_ID,
        service_principal_client_secret=SP_CLIENT_SECRET,
    )
else:
    vsc = VectorSearchClient()  # use notebook identity

index = vsc.get_index(endpoint_name=ENDPOINT_NAME, index_name=INDEX_NAME)
print(f"Connected to index: {INDEX_NAME}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. The unlock — `query_type="FULL_TEXT"`
# MAGIC
# MAGIC Databricks Vector Search supports three query modes on a hybrid index:
# MAGIC
# MAGIC | Mode | What it does | When to use |
# MAGIC |---|---|---|
# MAGIC | `ANN` | Pure semantic similarity via embeddings | Conceptual queries ("how do I ship a fragile item?") |
# MAGIC | `HYBRID` | ANN + full-text fused via RRF | General-purpose Q&A |
# MAGIC | `FULL_TEXT` | **Pure Okapi BM25, no embeddings** | Product names, codes, jargon — what we want here |
# MAGIC
# MAGIC FULL_TEXT mode is in beta but available on any hybrid index — no new endpoint, no new index.

# COMMAND ----------

# Quick raw test — what does FULL_TEXT mode return?
raw = index.similarity_search(
    query_text="ENTArgo",          # a known EL-domain term
    columns=["source"],
    num_results=20,
    query_type="FULL_TEXT",
)

print("Columns:", [c["name"] for c in raw["manifest"]["columns"]])
print(f"Rows returned: {raw['result']['row_count']}")
print("\nTop 10 raw hits (source, BM25 score):")
for row in raw["result"]["data_array"][:10]:
    print(f"  {row[0]:30s}  score={row[1]:.2f}")

# Expected: top hits clustered on `myNET` and `Network Manual EL` (EL-domain sources)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Aggregation — from raw hits to a routing decision
# MAGIC
# MAGIC The raw BM25 output gives us chunk-level hits. For routing, we want **domain-level signal**:
# MAGIC group hits by source/domain, sum BM25 scores. Summing rewards both *frequency* and *relevance*
# MAGIC simultaneously — exactly what we want for "which domain has the most + strongest presence of
# MAGIC this term."

# COMMAND ----------

# Maintained by the domain team — see open question (a) at the bottom.
SOURCE_TO_DOMAIN: dict[str, str] = {
    "myNET":             "european_logistics",
    "Network Manual EL": "european_logistics",
    # "ASLConnect":      "air_sea_logistics",
    # "FoodNet":         "food_logistics",
    # ... fill in as you discover them in your corpus
}


def term_locator(
    terms: list[str],
    top_k: int = 3,
    num_candidates: int = 50,
    group_by: str = "source",
) -> list[dict]:
    """Locate which sources/domains best match the given technical terms.

    Args:
        terms:           technical keywords extracted from the user query.
        top_k:           how many top groups to return.
        num_candidates:  BM25 hits to retrieve before aggregating (max 200).
        group_by:        column to group on — "source" today, "domain" once
                         we add that column to the source table (see open q. a).

    Returns:
        list of {"name", "domain", "score_sum", "hits"} sorted desc by score_sum.
    """
    if not terms:
        return []

    resp = index.similarity_search(
        query_text=" ".join(terms),
        columns=[group_by],
        num_results=num_candidates,
        query_type="FULL_TEXT",
    )
    rows = (resp.get("result") or {}).get("data_array") or []

    agg: dict[str, dict] = defaultdict(lambda: {"score_sum": 0.0, "hits": 0})
    for row in rows:
        name, score = row[0], row[-1]
        if name is None:
            continue
        agg[name]["score_sum"] += float(score)
        agg[name]["hits"]      += 1

    ranked = sorted(
        (
            {
                "name":      name,
                "domain":    SOURCE_TO_DOMAIN.get(name, "unknown"),
                "score_sum": round(stats["score_sum"], 2),
                "hits":      stats["hits"],
            }
            for name, stats in agg.items()
        ),
        key=lambda x: x["score_sum"],
        reverse=True,
    )
    return ranked[:top_k]

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Test cases — walk through with the team
# MAGIC
# MAGIC Run each cell, discuss what we see. Watch for:
# MAGIC - **Clear winner** — one domain dominates → easy routing decision
# MAGIC - **Ambiguous** — top two are close → meta-tool flags this, Supervisor handles
# MAGIC - **No hits** — term not indexed → Supervisor asks user to clarify, doesn't guess

# COMMAND ----------

# Case 1 — single known EL term
print("=== Case 1: ['Entago'] (single EL-only term) ===")
for r in term_locator(terms=["Entago"]):
    print(r)

# COMMAND ----------

# Case 2 — multi-term query (combines BM25 scores naturally)
print("=== Case 2: ['Entago', 'palletization'] ===")
for r in term_locator(terms=["Entago", "palletization"]):
    print(r)

# COMMAND ----------

# Case 3 — generic logistics term (should hit broadly; demonstrates the "don't call me for generics" case)
print("=== Case 3: ['shipment'] (generic — Supervisor should NOT call us for this) ===")
for r in term_locator(terms=["shipment"]):
    print(r)

# COMMAND ----------

# Case 4 — unknown term (should return empty → Supervisor asks user)
print("=== Case 4: ['ZxQzaa123'] (deliberately not in corpus) ===")
results = term_locator(terms=["ZxQzaa123"])
print(results if results else "(no hits — Supervisor should clarify with user)")

# COMMAND ----------

# Case 5 — add your own. Team: edit and re-run with terms that have bitten you in the past.
TERMS_TO_TRY = [
    # "EntagoConnect",
    # "ASLConnect",
    # "FoodNet groupage",
    # "<your domain-specific term>",
]
print(f"=== Case 5: {TERMS_TO_TRY} ===")
for r in term_locator(terms=TERMS_TO_TRY) if TERMS_TO_TRY else []:
    print(r)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. LLM-callable wrapper — what the Supervisor actually sees
# MAGIC
# MAGIC The raw `term_locator` returns Python dicts; the Supervisor needs a **string** that's easy to
# MAGIC reason over. The wrapper below:
# MAGIC - Formats the ranked list as human-readable lines
# MAGIC - Adds an **ambiguity warning** when top-2 are within 70% of each other — the Supervisor uses
# MAGIC   this to decide between "route confidently" vs "ask user to clarify" vs "query both"
# MAGIC - Explicit no-hit message so the Supervisor doesn't hallucinate

# COMMAND ----------

def term_locator_tool(terms: list[str]) -> str:
    """LLM-callable wrapper. Returns a human-readable summary the Supervisor
    can reason over for routing."""
    results = term_locator(terms=terms, top_k=3)

    if not results:
        return (
            f"No hits for terms {terms!r}. Likely not indexed or misspelled. "
            f"Ask user to clarify; do not guess."
        )

    lines = [f"Top candidates for terms {terms!r}:"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"  {i}. source='{r['name']}' (domain={r['domain']}), "
            f"score={r['score_sum']}, hits={r['hits']}"
        )

    top = results[0]
    runner_up = results[1] if len(results) > 1 else None
    if runner_up and runner_up["score_sum"] >= 0.7 * top["score_sum"]:
        lines.append(
            "⚠  Top two candidates are close — terms may be ambiguous. "
            "Consider querying both sub-agents or asking user to clarify."
        )
    else:
        lines.append(f"→ Recommended route: domain='{top['domain']}'")

    return "\n".join(lines)


# Walk these outputs through with the team — these are what the Supervisor's
# LLM sees and reasons over.
print(term_locator_tool(terms=["Entago"]))
print()
print(term_locator_tool(terms=["Entago", "ASLConnect"]))   # likely ambiguous
print()
print(term_locator_tool(terms=["totally_unknown_term"]))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Supervisor tool description (copy into AgentBricks)
# MAGIC
# MAGIC This is the **prompt-engineering surface** for the Supervisor — when it calls this tool, when
# MAGIC it doesn't, and how to interpret the output. Drop this into the `instructions` field when
# MAGIC registering `term_locator` as a Supervisor sub-tool.
# MAGIC
# MAGIC ```text
# MAGIC TOOL: term_locator
# MAGIC
# MAGIC PURPOSE
# MAGIC   Routing-helper. Returns WHERE in the knowledge base technical terms appear,
# MAGIC   NOT the answer. Call BEFORE routing to a domain sub-agent when terms are
# MAGIC   unfamiliar.
# MAGIC
# MAGIC WHEN TO CALL  (any one of)
# MAGIC   · Query contains a likely product name (capitalized, non-generic, e.g. "Entago", "myNET").
# MAGIC   · Query contains an internal code or acronym not in {EL, ASL, FL, SLA, KPI, ETA, ETD}.
# MAGIC   · You cannot pick a single domain with high confidence from tool descriptions alone.
# MAGIC
# MAGIC INPUT
# MAGIC   terms : list[str] — the technical / unfamiliar tokens extracted from the
# MAGIC                        user query (NOT the full query).
# MAGIC
# MAGIC OUTPUT
# MAGIC   Ranked candidates with cumulative BM25 scores, hit counts, and an explicit
# MAGIC   ambiguity warning if the top two are close.
# MAGIC
# MAGIC WHEN NOT TO CALL
# MAGIC   · Conversational turn ("hi", "thanks", "what can you do?").
# MAGIC   · All terms are generic ("shipment status", "delivery rate", "ETA").
# MAGIC   · Domain context already established in the previous turn.
# MAGIC
# MAGIC INTERPRETING THE RESULT
# MAGIC   · Clear winner    → route to that sub-agent.
# MAGIC   · Ambiguity flag  → query both, or ask user to clarify which context.
# MAGIC   · No hits         → ask user to clarify. Do not guess.
# MAGIC ```

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Open design decisions — discussion points for Session 11
# MAGIC
# MAGIC ### (a) `source` vs `domain` as the grouping column
# MAGIC The script groups on `source` and maps to domain via the `SOURCE_TO_DOMAIN` dict. Cleaner long-term:
# MAGIC add a `domain` column to the source Delta table backing the VS index, group on it directly,
# MAGIC retire the dict. **Decision for the team:** does the same `source` value ever cross domains? If
# MAGIC never → add the column.
# MAGIC
# MAGIC ### (b) Term-extraction quality
# MAGIC The Supervisor's LLM extracts `terms` from the user query. Worth measuring: how often does it
# MAGIC pull the *right* terms vs miss them or include generic noise? Easy to wire as a scorer on the
# MAGIC existing golden dataset.
# MAGIC
# MAGIC ### (c) Spelling / variant resilience
# MAGIC Databricks BM25 is exact-token (word-boundary split, lowercased, punctuation stripped).
# MAGIC `Entago` matches `Entago`, but **not** `Entago®` if the trademark symbol attaches without
# MAGIC whitespace, and **not** `Entargo`. Pick one:
# MAGIC - **Index-time normalization** (canonicalize variants in the source table) — long-term winner
# MAGIC - **Query-time variant expansion** — cheaper but doesn't help users who type new variants
# MAGIC
# MAGIC ### (d) Ambiguity threshold tuning
# MAGIC The `0.7 × top_score` ambiguity rule is a starting point. Abdullah's test outputs (Case 5
# MAGIC above) will give us a real score distribution to pick a defensible threshold — probably
# MAGIC somewhere in the 0.5–0.8 range.
# MAGIC
# MAGIC ### (e) Cost guard
# MAGIC Each `term_locator` call ≈ 50–100 ms of VS query. Cheap, but if the Supervisor over-calls it
# MAGIC (e.g. on every conversational turn), it shows up in MLflow traces. The tool description above
# MAGIC discourages reflex use. Monitor in production.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Next steps after Session 11
# MAGIC
# MAGIC 1. **Abdullah** runs Case 5 with 5–10 real jargon queries, sends back the raw outputs.
# MAGIC 2. **Regina** confirms `source → domain` mapping (or signs off on adding a `domain` column to the source table).
# MAGIC 3. **Badr** ships a deterministic eval scorer ("did `term_locator` return the correct domain on labeled cases?")
# MAGIC    using the golden dataset — no LLM judge needed, fast feedback loop.
# MAGIC 4. Once routing accuracy looks good in the eval, register `term_locator` as a UC function and add it
# MAGIC    to the Supervisor as a sub-tool with the prompt from Section 6.
# MAGIC
# MAGIC ### References
# MAGIC - Databricks docs: [Query a vector search index — FULL_TEXT mode](https://docs.databricks.com/aws/en/vector-search/query-vector-search)
# MAGIC - Session 10 origin: §5 "Supervisor routing on domain-specific terminology"
# MAGIC - Literature alignment: Tool-to-Agent Retrieval (arXiv 2511.01854) — same hybrid BM25 + dense pattern, +19.4% Recall@5 over prior SOTA
