# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # 06 — Genie Space Setup
# MAGIC
# MAGIC **Creates:** Synthetic `project_tracker` Delta table + instructions for Genie Space
# MAGIC
# MAGIC **Time:** ~2 minutes | **Cost:** Free
# MAGIC
# MAGIC ## What is Genie?
# MAGIC
# MAGIC Genie (AI/BI) lets users ask natural language questions about structured data.
# MAGIC It translates questions to SQL, executes them, and returns formatted results.
# MAGIC
# MAGIC ```
# MAGIC User: "Which projects are over budget?"
# MAGIC     │
# MAGIC     ▼  Genie (NL → SQL)
# MAGIC SELECT * FROM project_tracker WHERE actual_cost > budget
# MAGIC     │
# MAGIC     ▼  SQL Warehouse
# MAGIC ┌──────────────┬────────┬────────────┬───────┐
# MAGIC │ project_name │ budget │ actual_cost│ delta │
# MAGIC ├──────────────┼────────┼────────────┼───────┤
# MAGIC │ Data Lake    │ 50000  │ 62000      │+12000 │
# MAGIC └──────────────┴────────┴────────────┴───────┘
# MAGIC ```
# MAGIC
# MAGIC Genie is a native Supervisor sub-agent type — no wrapping needed.

# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1: Create Synthetic Project Tracker Table
# MAGIC
# MAGIC A realistic-looking table that demonstrates Genie's SQL generation capabilities.
# MAGIC Columns designed to trigger interesting queries (budget vs actual, status filters,
# MAGIC date ranges, team-based aggregations).

# COMMAND ----------
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType
from datetime import date

schema = StructType([
    StructField("project_id", IntegerType(), False),
    StructField("project_name", StringType(), False),
    StructField("team", StringType(), False),
    StructField("owner", StringType(), False),
    StructField("status", StringType(), False),
    StructField("priority", StringType(), False),
    StructField("budget", IntegerType(), False),
    StructField("actual_cost", IntegerType(), False),
    StructField("start_date", DateType(), False),
    StructField("target_date", DateType(), False),
    StructField("technology", StringType(), False),
])

data = [
    (1,  "Customer 360 Platform",     "Data Engineering", "Alice Chen",    "In Progress", "High",   120000, 95000,  date(2025, 9, 1),  date(2026, 6, 30), "Databricks + Delta Lake"),
    (2,  "ML Fraud Detection",        "Data Science",     "Bob Martinez",  "In Progress", "High",   85000,  72000,  date(2025, 11, 1), date(2026, 4, 30), "MLflow + Feature Store"),
    (3,  "Real-time Inventory Sync",  "Data Engineering", "Carol Weber",   "Completed",   "Medium", 45000,  43000,  date(2025, 6, 1),  date(2025, 12, 31), "Kafka + Delta Live Tables"),
    (4,  "HR Analytics Dashboard",    "Analytics",        "David Kim",     "In Progress", "Low",    30000,  28000,  date(2026, 1, 15), date(2026, 5, 31), "SQL Warehouse + Dashboard"),
    (5,  "Supply Chain Optimization", "Data Science",     "Elena Rossi",   "At Risk",     "High",   200000, 185000, date(2025, 7, 1),  date(2026, 3, 31), "LangGraph + Databricks Apps"),
    (6,  "Data Governance Rollout",   "Platform",         "Frank Liu",     "In Progress", "High",   60000,  35000,  date(2026, 2, 1),  date(2026, 8, 31), "Unity Catalog + Lineage"),
    (7,  "Chat Support Bot",          "AI/ML",            "Grace Tanaka",  "Planning",    "Medium", 75000,  5000,   date(2026, 4, 1),  date(2026, 9, 30), "AgentBricks + Teams Bot"),
    (8,  "ETL Migration (Legacy)",    "Data Engineering", "Henry Müller",  "At Risk",     "High",   150000, 162000, date(2025, 3, 1),  date(2026, 1, 31), "Delta Live Tables + Workflows"),
    (9,  "Predictive Maintenance",    "Data Science",     "Ines Durand",   "Completed",   "Medium", 90000,  88000,  date(2025, 4, 1),  date(2025, 11, 30), "Spark ML + Model Serving"),
    (10, "Customer Churn Model",      "Data Science",     "James O'Brien", "In Progress", "High",   55000,  48000,  date(2025, 12, 1), date(2026, 5, 31), "AutoML + Feature Store"),
    (11, "Compliance Reporting",      "Analytics",        "Karen Patel",   "In Progress", "Medium", 40000,  22000,  date(2026, 1, 1),  date(2026, 7, 31), "SQL Warehouse + Lakeview"),
    (12, "Edge IoT Pipeline",         "Data Engineering", "Liam Schmidt",  "Planning",    "Low",    95000,  8000,   date(2026, 5, 1),  date(2026, 12, 31), "Structured Streaming + UC"),
    (13, "Document Search Agent",     "AI/ML",            "Maria Lopez",   "In Progress", "High",   35000,  20000,  date(2026, 3, 1),  date(2026, 6, 30), "LangGraph + Vector Search"),
    (14, "Sales Forecasting v2",      "Data Science",     "Nils Eriksson", "Completed",   "Medium", 65000,  61000,  date(2025, 5, 1),  date(2025, 10, 31), "Prophet + Model Serving"),
    (15, "API Gateway Migration",     "Platform",         "Olivia Park",   "At Risk",     "High",   110000, 125000, date(2025, 8, 1),  date(2026, 2, 28), "Kong + Azure API Mgmt"),
    (16, "Self-Service BI Portal",    "Analytics",        "Peter Novak",   "Planning",    "Medium", 50000,  3000,   date(2026, 6, 1),  date(2026, 11, 30), "Genie + Lakeview Dashboards"),
    (17, "Data Quality Framework",    "Platform",         "Qi Zhang",      "In Progress", "High",   70000,  45000,  date(2025, 10, 1), date(2026, 4, 30), "UC Monitors + Expectations"),
    (18, "Marketing Attribution",     "Analytics",        "Rita Sharma",   "In Progress", "Low",    25000,  18000,  date(2026, 2, 15), date(2026, 7, 31), "SQL + Dashboard"),
    (19, "Knowledge Base Agent",      "AI/ML",            "Stefan Braun",  "In Progress", "High",   80000,  55000,  date(2026, 1, 1),  date(2026, 6, 30), "AgentBricks + Supervisor"),
    (20, "Cost Optimization Engine",  "Platform",         "Tanya Volkov",  "Planning",    "Medium", 45000,  2000,   date(2026, 7, 1),  date(2027, 1, 31), "Workflows + Budget Policies"),
]

df = spark.createDataFrame(data, schema=schema)
df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(TABLE_GENIE)

print(f"✓ Created {TABLE_GENIE} with {df.count()} rows")
df.show(truncate=40)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2: Verify Table for Genie
# MAGIC
# MAGIC Run some sample queries to verify the data supports interesting Genie conversations.

# COMMAND ----------
# Sample queries Genie should handle well:
print("=== Sample Queries for Genie ===\n")

print("1. 'Which projects are over budget?'")
spark.sql(f"""
    SELECT project_name, team, budget, actual_cost, actual_cost - budget as over_by
    FROM {TABLE_GENIE} WHERE actual_cost > budget ORDER BY over_by DESC
""").show(truncate=40)

print("2. 'What is each team's total budget?'")
spark.sql(f"""
    SELECT team, COUNT(*) as projects, SUM(budget) as total_budget, SUM(actual_cost) as total_spent
    FROM {TABLE_GENIE} GROUP BY team ORDER BY total_budget DESC
""").show(truncate=40)

print("3. 'Which high priority projects are at risk?'")
spark.sql(f"""
    SELECT project_name, owner, status, budget, actual_cost
    FROM {TABLE_GENIE} WHERE priority = 'High' AND status = 'At Risk'
""").show(truncate=40)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3: Create Genie Space (UI — manual step)
# MAGIC
# MAGIC Genie Spaces are created via the workspace UI. Follow these steps:
# MAGIC
# MAGIC 1. **Navigate:** Sidebar → **Genie** (or search "Genie" in the top bar)
# MAGIC 2. **Click:** "New Genie Space"
# MAGIC 3. **Configure:**
# MAGIC    - **Name:** `Project Tracker Genie`
# MAGIC    - **SQL Warehouse:** Select your serverless warehouse
# MAGIC    - **Tables:** Add `my_catalog.agent_lab.project_tracker`
# MAGIC    - **Instructions** (optional but recommended):
# MAGIC      ```
# MAGIC      You help users explore project portfolio data. The table contains
# MAGIC      project tracking data with budgets, timelines, teams, and statuses.
# MAGIC      When users ask about budgets, compare actual_cost vs budget.
# MAGIC      When users ask about risks, filter by status = 'At Risk'.
# MAGIC      ```
# MAGIC 4. **Save** and test with: "Which projects are over budget?"
# MAGIC
# MAGIC ### After creating, note the Genie Space ID:
# MAGIC - URL will look like: `https://<workspace>/genie/rooms/<GENIE_ID>`
# MAGIC - Copy the `<GENIE_ID>` — you'll need it for notebook 07 (Supervisor)

# COMMAND ----------
# After creating Genie Space, paste its ID here for reference:
GENIE_SPACE_ID = ""  # ← Paste your Genie Space ID here after creating it in the UI

if GENIE_SPACE_ID:
    print(f"✓ Genie Space ID saved: {GENIE_SPACE_ID}")
else:
    print("⚠ Create the Genie Space in the UI, then paste the ID above")
    print("  URL pattern: https://<workspace>/genie/rooms/<ID>")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4: Test Genie via SDK (optional)

# COMMAND ----------
if GENIE_SPACE_ID:
    # Start a Genie conversation via SDK
    genie_resp = _w.genie.start_conversation(
        space_id=GENIE_SPACE_ID,
        content="Which projects are over budget?",
    )
    print(f"Conversation ID: {genie_resp.conversation_id}")
    print(f"Message ID: {genie_resp.message_id}")

    # Poll for result
    import time
    for i in range(10):
        msg = _w.genie.get_message(
            space_id=GENIE_SPACE_ID,
            conversation_id=genie_resp.conversation_id,
            message_id=genie_resp.message_id,
        )
        status = msg.status
        print(f"  [{i}] Status: {status}")
        if status and status.value in ("COMPLETED", "FAILED"):
            break
        time.sleep(3)

    # Print results
    if msg.attachments:
        for att in msg.attachments:
            if att.text:
                print(f"\nGenie says: {att.text.content[:500]}")
            if att.query:
                print(f"\nSQL: {att.query.query}")
else:
    print("Skip — set GENIE_SPACE_ID first")
