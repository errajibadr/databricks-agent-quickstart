# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 07 — Supervisor Agent
# MAGIC
# MAGIC **Creates:** Supervisor Agent with sub-agents (KA + UC Tool + Genie)
# MAGIC
# MAGIC **Time:** ~2 minutes | **Cost:** dbu per usage (Supervisor is a managed service)
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

UC_TOOL_FUNCTION

# COMMAND ----------

# ═══════════════════════════════════════════════════════════
#  FILL IN YOUR SUB-AGENT IDS
# ═══════════════════════════════════════════════════════════

# From notebook 05: UC Function (always available if you ran 04+05)
UC_TOOL_NAME = UC_TOOL_FUNCTION  # e.g., "my_catalog.agent_lab.ask_doc_agent"

# From notebook 06: Genie Space ID (from the UI URL)
GENIE_SPACE_ID = "01f14336dc401a758b46a491256a9026"  # ← Paste your Genie Space ID

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

sub_agents = []

# Sub-agent 1: UC Function (your custom LangGraph agent)
sub_agents.append({
    "name": "doc_search_agent",
    "description": (
        "Searches LangChain documentation using a custom LangGraph agent with "
        "Vector Search. Use this when the user asks about LangChain concepts, "
        "APIs, tool calling, RAG, agents, chains, memory, or related topics."
    ),
    "type": "UC_FUNCTION",
    "uc_function_name": UC_TOOL_NAME,
})

# Sub-agent 2: Genie Space (project tracker)
if GENIE_SPACE_ID:
    sub_agents.append({
        "name": "project_tracker",
        "description": (
            "Queries project portfolio data including budgets, timelines, teams, "
            "and statuses. Use this when the user asks about project costs, "
            "team allocations, deadlines, at-risk projects, or budget analysis."
        ),
        "type": "GENIE",
        "genie_space_id": GENIE_SPACE_ID,
    })

# Sub-agent 3: Knowledge Assistant (optional)
if KA_ID:
    sub_agents.append({
        "name": "knowledge_assistant",
        "description": (
            "General-purpose knowledge assistant for documentation and knowledge base "
            "queries. Use this for broad information requests not covered by other agents."
        ),
        "type": "KNOWLEDGE_ASSISTANT",
        "knowledge_assistant_id": KA_ID,
    })

print(f"Configured {len(sub_agents)} sub-agents:")
for sa in sub_agents:
    print(f"  - {sa['name']} ({sa['type']})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Create Supervisor via REST API
# MAGIC
# MAGIC The Supervisor API is REST-only — no SDK support yet.
# MAGIC We use the workspace token from `WorkspaceClient` for auth.

# COMMAND ----------

workspace_url = _w.config.host.rstrip("/")
token = _w.config.tok
print(f"Workspace URL: {workspace_url}")
print(f"Token: {token}")

# COMMAND ----------

import requests

# Get workspace URL and token from the SDK client
workspace_url = _w.config.host.rstrip("/")

headers = {
    **_w.config.authenticate(),
    "Content-Type": "application/json",
}

# Build Supervisor payload
supervisor_payload = {
    "name": SUPERVISOR_NAME,
    "description": (
        "Workspace Kit Supervisor — routes questions to specialized agents: "
        "doc search (LangChain docs), project tracker (Genie), "
        "and knowledge assistant (if available)."
    ),
    "sub_agents": sub_agents,
    "system_prompt": (
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

headers

# COMMAND ----------

# Create or update the Supervisor
api_url = f"{workspace_url}/api/2.0/multi-agent-supervisors"

# Check if supervisor already exists
list_resp = requests.get(api_url, headers=headers)
existing = None
if list_resp.status_code == 200:
    for s in list_resp.json().get("supervisors", []):
        if s.get("name") == SUPERVISOR_NAME:
            existing = s
            break

if existing:
    # Update existing
    supervisor_id = existing["supervisor_id"]
    update_resp = requests.put(
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
        supervisor_id = result.get("supervisor_id", "unknown")
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
if 'supervisor_id' in dir() and supervisor_id:
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
