# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # 07 — Supervisor Agent
# MAGIC
# MAGIC **Creates:** Supervisor Agent with sub-agents (KA + UC Tool + Genie)
# MAGIC
# MAGIC **Time:** ~2 minutes | **Cost:** Free (Supervisor is a managed service)
# MAGIC
# MAGIC ## Architecture
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────┐
# MAGIC │                  SUPERVISOR AGENT                       │
# MAGIC │         (routes user questions to sub-agents)           │
# MAGIC │                                                         │
# MAGIC │  "What is tool calling?"                                │
# MAGIC │       │                                                 │
# MAGIC │       ▼  Supervisor LLM decides routing                │
# MAGIC │                                                         │
# MAGIC │  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  │
# MAGIC │  │ UC Function  │  │ Genie Space  │  │ Knowledge    │  │
# MAGIC │  │ (doc agent)  │  │ (projects)   │  │ Assistant    │  │
# MAGIC │  │              │  │              │  │ (if avail.)  │  │
# MAGIC │  │ ask_doc_agent│  │ project data │  │ general docs │  │
# MAGIC │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
# MAGIC │         │                 │                  │          │
# MAGIC │         ▼                 ▼                  ▼          │
# MAGIC │  Serving Endpoint   SQL Warehouse      VS Index        │
# MAGIC └─────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC **Supervisor API:** REST-only (`/api/2.0/multi-agent-supervisors`).
# MAGIC No SDK support yet. We use `requests` via the workspace token.

# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1: Configuration — Define Sub-Agents
# MAGIC
# MAGIC Edit the IDs below based on what you created in previous notebooks.

# COMMAND ----------
# ═══════════════════════════════════════════════════════════
#  FILL IN YOUR SUB-AGENT IDS
# ═══════════════════════════════════════════════════════════

# From notebook 05: UC Function (always available if you ran 04+05)
UC_TOOL_NAME = UC_TOOL_FUNCTION  # e.g., "my_catalog.agent_lab.ask_doc_agent"

# From notebook 06: Genie Space ID (from the UI URL)
GENIE_SPACE_ID = ""  # ← Paste your Genie Space ID

# Optional: Knowledge Assistant ID (if KA is enabled on your workspace)
# Create a KA in the UI first, then paste its ID here
KA_ID = ""  # ← Paste your KA ID if available

print(f"UC Tool:  {UC_TOOL_NAME}")
print(f"Genie:    {GENIE_SPACE_ID or '(not set)'}")
print(f"KA:       {KA_ID or '(not set)'}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2: Build Sub-Agent Definitions

# COMMAND ----------
import json

# Each agent in the `agents` list carries `agent_type` plus the matching
# nested config block. This is the canonical REST shape (what the
# `/api/2.0/multi-agent-supervisors` endpoint actually accepts). The
# `manage_mas` MCP tool exposes a flatter convenience surface (e.g.
# `uc_function_name`, `ka_tile_id`, `genie_space_id`) — it then translates
# those into this nested form before posting. We're calling REST directly,
# so we use the nested form.
agents = []

# Sub-agent 1: UC Function (your custom LangGraph agent)
# UC_TOOL_NAME is "catalog.schema.function_name"; split into uc_path parts.
_uc_parts = UC_TOOL_NAME.split(".")
assert len(_uc_parts) == 3, f"UC_TOOL_NAME must be 'catalog.schema.function', got: {UC_TOOL_NAME}"
agents.append(
    {
        "name": "doc_search_agent",
        "description": (
            "Searches LangChain documentation using a custom LangGraph agent with "
            "Vector Search. Use this when the user asks about LangChain concepts, "
            "APIs, tool calling, RAG, agents, chains, memory, or related topics."
        ),
        "agent_type": "unity_catalog_function",
        "unity_catalog_function": {
            "uc_path": {
                "catalog": _uc_parts[0],
                "schema": _uc_parts[1],
                "name": _uc_parts[2],
            }
        },
    }
)

# Sub-agent 2: Genie Space (project tracker)
if GENIE_SPACE_ID:
    agents.append(
        {
            "name": "project_tracker",
            "description": (
                "Queries project portfolio data including budgets, timelines, teams, "
                "and statuses. Use this when the user asks about project costs, "
                "team allocations, deadlines, at-risk projects, or budget analysis."
            ),
            "agent_type": "genie",
            "genie_space": {"id": GENIE_SPACE_ID},
        }
    )

# Sub-agent 3: Knowledge Assistant (optional)
# KAs are addressed via their serving endpoint, not via the tile_id directly.
# The endpoint name is `ka-<first-segment-of-tile-id>-endpoint`.
if KA_ID:
    _ka_endpoint = f"ka-{KA_ID.split('-')[0]}-endpoint"
    agents.append(
        {
            "name": "knowledge_assistant",
            "description": (
                "General-purpose knowledge assistant for documentation and knowledge base "
                "queries. Use this for broad information requests not covered by other agents."
            ),
            "agent_type": "serving_endpoint",
            "serving_endpoint": {"name": _ka_endpoint},
        }
    )

print(f"Configured {len(agents)} sub-agents:")
for a in agents:
    print(f"  - {a['name']} ({a['agent_type']})")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3: Create Supervisor via REST API
# MAGIC
# MAGIC The Supervisor API is REST-only — no SDK support yet.
# MAGIC We use the workspace token from `WorkspaceClient` for auth.

# COMMAND ----------
import requests

# Get workspace URL + auth headers from the SDK client.
# `_w.config.authenticate()` returns the auth header dict dynamically and
# works for OAuth (notebook default), PAT, and service-principal chains.
# Don't read `_w.config.token` directly: it's None under OAuth because OAuth
# doesn't expose a static bearer — the token is computed per-request.
workspace_url = _w.config.host.rstrip("/")
headers = {
    **_w.config.authenticate(),
    "Content-Type": "application/json",
}

# Canonical Supervisor payload — top-level keys are `name`, `agents`,
# `description`, `instructions`. Previous `sub_agents` / `system_prompt`
# names are not recognized by the API and would be silently dropped.
supervisor_payload = {
    "name": SUPERVISOR_NAME,
    "description": (
        "Workspace Kit Supervisor — routes questions to specialized agents: "
        "doc search (LangChain docs), project tracker (Genie), "
        "and knowledge assistant (if available)."
    ),
    "agents": agents,
    "instructions": (
        "You are a helpful supervisor agent. Route user questions to the most "
        "appropriate sub-agent based on the topic:\n"
        "- LangChain docs, APIs, or code → doc_search_agent\n"
        "- Project data, budgets, teams, timelines → project_tracker\n"
        "- General knowledge questions → knowledge_assistant (if available)\n"
        "If unsure, ask the user to clarify which domain their question is about."
    ),
}

print("Supervisor payload:")
print(json.dumps(supervisor_payload, indent=2))

# COMMAND ----------
# Create or update the Supervisor
api_url = f"{workspace_url}/api/2.0/multi-agent-supervisors"

# Look up an existing Supervisor by name via the canonical tiles endpoint
# (filtered by tile_type=MAS). There is no `list` op on `/multi-agent-supervisors`
# itself — `mas_find_by_name` in the Databricks AgentBricks Manager source is
# the reference pattern.
tiles_url = f"{workspace_url}/api/2.0/tiles"
list_resp = requests.get(
    tiles_url,
    headers=headers,
    params={"filter": f"name_contains={SUPERVISOR_NAME}&&tile_type=MAS"},
)
existing = None
if list_resp.status_code == 200:
    for tile in list_resp.json().get("tiles", []):
        if tile.get("name") == SUPERVISOR_NAME:
            existing = tile
            break

if existing:
    # Update existing — verb is PATCH per the proto-canonical Manager source.
    supervisor_id = existing["tile_id"]
    update_resp = requests.patch(
        f"{api_url}/{supervisor_id}",
        headers=headers,
        json=supervisor_payload,
    )
    if update_resp.status_code == 200:
        print(f"✓ Updated Supervisor: {supervisor_id}")
    else:
        print(f"✗ Update failed: {update_resp.status_code} {update_resp.text}")
else:
    # Create new
    create_resp = requests.post(
        api_url,
        headers=headers,
        json=supervisor_payload,
    )
    if create_resp.status_code == 200:
        result = create_resp.json()
        # Response shape: {"multi_agent_supervisor": {"tile": {"tile_id": "..."}, ...}}
        supervisor_id = (
            result.get("multi_agent_supervisor", {}).get("tile", {}).get("tile_id", "unknown")
        )
        print(f"✓ Created Supervisor: {supervisor_id}")
    else:
        print(f"✗ Create failed: {create_resp.status_code} {create_resp.text}")
        print("  If 'feature not enabled', check that AgentBricks is enabled on this workspace")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4: Test Supervisor
# MAGIC
# MAGIC Two test paths:
# MAGIC 1. **AI Playground (recommended):** Go to AI Playground → select your Supervisor → chat
# MAGIC 2. **REST API (below):** Programmatic test

# COMMAND ----------
# Test via REST API
if "supervisor_id" in dir() and supervisor_id:
    test_url = f"{workspace_url}/api/2.0/multi-agent-supervisors/{supervisor_id}/chat"

    # Test 1: Should route to doc_search_agent (UC function)
    test1 = requests.post(
        test_url,
        headers=headers,
        json={"messages": [{"role": "user", "content": "What is tool calling in LangChain?"}]},
    )
    print("=== Test 1: LangChain question (→ doc_search_agent) ===")
    if test1.status_code == 200:
        print(json.dumps(test1.json(), indent=2)[:500])
    else:
        print(f"  Error: {test1.status_code} {test1.text[:200]}")

    # Test 2: Should route to project_tracker (Genie)
    if GENIE_SPACE_ID:
        test2 = requests.post(
            test_url,
            headers=headers,
            json={"messages": [{"role": "user", "content": "Which projects are over budget?"}]},
        )
        print("\n=== Test 2: Project question (→ project_tracker) ===")
        if test2.status_code == 200:
            print(json.dumps(test2.json(), indent=2)[:500])
        else:
            print(f"  Error: {test2.status_code} {test2.text[:200]}")
else:
    print("Skip — Supervisor not created yet")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. **Test in AI Playground** — the most natural way to interact with Supervisor
# MAGIC 2. **Run notebook 08** — comparative evaluation (Supervisor vs KA vs custom agent)
# MAGIC 3. **When done** — run notebook 99_cleanup to tear down resources
