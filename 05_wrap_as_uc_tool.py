# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # 05 — Wrap Agent as UC Function (for Supervisor)
# MAGIC
# MAGIC **Creates:** A Unity Catalog SQL function that calls your serving endpoint
# MAGIC
# MAGIC **Time:** ~1 minute | **Cost:** Free (just metadata)
# MAGIC
# MAGIC ## Why This Matters
# MAGIC
# MAGIC Supervisor Agent only supports these sub-agent types:
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────┐
# MAGIC │  Supervisor Sub-Agent Types                  │
# MAGIC │                                              │
# MAGIC │  1. Knowledge Assistant (KA)                 │
# MAGIC │  2. Genie Space                              │
# MAGIC │  3. UC Function  ◄── THIS IS YOUR ESCAPE    │
# MAGIC │  4. Agent (serving endpoint, KA-style)       │
# MAGIC │  5. MCP Tool                                 │
# MAGIC └─────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC Your custom LangGraph agent is deployed as a serving endpoint (type 4),
# MAGIC but Supervisor's "Agent" type expects KA-style registration.
# MAGIC The workaround: wrap your endpoint in a **UC Function** (type 3).
# MAGIC
# MAGIC `ai_query()` is a Databricks built-in SQL function that can call any
# MAGIC serving endpoint. Wrap it in a UC function → Supervisor sees it as a tool.
# MAGIC
# MAGIC ```
# MAGIC Supervisor ──► UC Function (ask_doc_agent)
# MAGIC                    │
# MAGIC                    ▼  ai_query()
# MAGIC               Serving Endpoint (langgraph-doc-agent)
# MAGIC                    │
# MAGIC                    ▼  LangGraph ReAct loop
# MAGIC               Vector Search ──► LLM ──► Response
# MAGIC ```

# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1: Create the UC Function
# MAGIC
# MAGIC `ai_query(endpoint, question)` sends the question to your serving endpoint
# MAGIC and returns the text response. The function's COMMENT becomes the tool
# MAGIC description that Supervisor reads to decide when to route to this tool.

# COMMAND ----------
spark.sql(f"""
CREATE OR REPLACE FUNCTION {UC_TOOL_FUNCTION}(
    question STRING COMMENT 'A question about LangChain documentation, APIs, patterns, or best practices'
)
RETURNS STRING
COMMENT 'Queries a LangGraph agent that searches LangChain documentation via Vector Search. Use this tool when the user asks about LangChain concepts, APIs, tool calling, RAG, agents, chains, memory, or related topics.'
RETURN SELECT ai_query(
    '{AGENT_ENDPOINT_NAME}',
    question
)
""")

print(f"✓ Created UC function: {UC_TOOL_FUNCTION}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2: Test the UC Function
# MAGIC
# MAGIC Call it via SQL to verify it routes to the serving endpoint correctly.

# COMMAND ----------
result = spark.sql(f"""
    SELECT {UC_TOOL_FUNCTION}('What is a retrieval chain in LangChain?') AS answer
""").collect()

print("=== UC Function Test ===")
print(result[0]["answer"][:500])

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3: Verify Function Metadata
# MAGIC
# MAGIC Check that the COMMENT (tool description) is set correctly — Supervisor
# MAGIC uses this to decide when to route to the function.

# COMMAND ----------
spark.sql(f"DESCRIBE FUNCTION EXTENDED {UC_TOOL_FUNCTION}").show(truncate=False)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4: Grant Permissions (for Supervisor)
# MAGIC
# MAGIC Supervisor's service principal needs EXECUTE permission on this function.
# MAGIC Also grant to any users who will interact with the Supervisor.

# COMMAND ----------
# Grant to all account users (adjust for production)
# spark.sql(f"GRANT EXECUTE ON FUNCTION {UC_TOOL_FUNCTION} TO `account users`")
# print("✓ Granted EXECUTE to account users")

# For specific users/groups:
# spark.sql(f"GRANT EXECUTE ON FUNCTION {UC_TOOL_FUNCTION} TO `user@example.com`")

print("⚠ Uncomment the GRANT statements above if Supervisor can't access the function")
print(f"\n✓ UC function ready: {UC_TOOL_FUNCTION}")
print("  Proceed to notebook 06 (Genie) or 07 (Supervisor)")
