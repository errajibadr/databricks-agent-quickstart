# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # 99 — Cleanup: Tear Down Everything
# MAGIC
# MAGIC **Deletes:** Serving endpoints, VS indexes, VS endpoint, tables, UC functions
# MAGIC
# MAGIC **Time:** ~2 minutes
# MAGIC
# MAGIC **Run this when you're done testing!** VS endpoints bill 24/7.
# MAGIC Serving endpoints with scale-to-zero are $0 when idle, but still consume
# MAGIC namespace. Clean up to avoid surprise charges.
# MAGIC
# MAGIC ```
# MAGIC ⚠  TEARDOWN ORDER MATTERS
# MAGIC ────────────────────────
# MAGIC 1. Delete Supervisor (REST API)
# MAGIC 2. Delete Serving Endpoints (agents)
# MAGIC 3. Delete VS Indexes (must be before VS endpoint!)
# MAGIC 4. Delete VS Endpoint (24/7 billing stops here)
# MAGIC 5. Delete UC Functions
# MAGIC 6. Delete Delta Tables
# MAGIC 7. Delete Volume contents (optional)
# MAGIC
# MAGIC Why this order: VS indexes become orphaned if you delete the endpoint first.
# MAGIC Serving endpoints should be deleted before the VS index they depend on.
# MAGIC ```

# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
# MAGIC %md
# MAGIC ## Safety Check

# COMMAND ----------
# ═══════════════════════════════════════════════════════════
#  SET TO True TO CONFIRM DELETION
# ═══════════════════════════════════════════════════════════
CONFIRM_DELETE = False  # ← Change to True to enable deletion
# ═══════════════════════════════════════════════════════════

if not CONFIRM_DELETE:
    print("⚠ CONFIRM_DELETE is False — nothing will be deleted.")
    print("  Set CONFIRM_DELETE = True and re-run to tear down resources.")
    dbutils.notebook.exit("Aborted — CONFIRM_DELETE is False")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1: Delete Supervisor

# COMMAND ----------
import requests
import json

workspace_url = _w.config.host.rstrip("/")
# `_w.config.authenticate()` returns auth headers dynamically — works for
# OAuth notebook auth (where `.config.token` would be None) and PAT alike.
headers = {**_w.config.authenticate(), "Content-Type": "application/json"}

# Find and delete supervisor
list_resp = requests.get(f"{workspace_url}/api/2.0/multi-agent-supervisors", headers=headers)
if list_resp.status_code == 200:
    for s in list_resp.json().get("supervisors", []):
        if s.get("name") == SUPERVISOR_NAME:
            sid = s["supervisor_id"]
            del_resp = requests.delete(f"{workspace_url}/api/2.0/multi-agent-supervisors/{sid}", headers=headers)
            print(f"{'✓' if del_resp.status_code == 200 else '✗'} Supervisor '{SUPERVISOR_NAME}' (id={sid})")
            break
    else:
        print(f"  Supervisor '{SUPERVISOR_NAME}' not found — already deleted?")
else:
    print(f"  Could not list supervisors: {list_resp.status_code}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2: Delete Serving Endpoint (Agent)

# COMMAND ----------
try:
    _w.serving_endpoints.delete(AGENT_ENDPOINT_NAME)
    print(f"✓ Deleted serving endpoint: {AGENT_ENDPOINT_NAME}")
except Exception as e:
    print(f"  Serving endpoint '{AGENT_ENDPOINT_NAME}' not found: {e}")

# Also delete the review endpoint created by agents.deploy()
try:
    _w.serving_endpoints.delete(f"{AGENT_ENDPOINT_NAME}-review")
    print(f"✓ Deleted review endpoint: {AGENT_ENDPOINT_NAME}-review")
except Exception:
    pass  # Review endpoint may not exist

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3: Delete VS Indexes (BEFORE endpoint!)

# COMMAND ----------
try:
    _w.vector_search_indexes.delete_index(VS_INDEX_NAME)
    print(f"✓ Deleted VS index: {VS_INDEX_NAME}")
except Exception as e:
    print(f"  VS index '{VS_INDEX_NAME}' not found: {e}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4: Delete VS Endpoint (stops 24/7 billing)

# COMMAND ----------
try:
    _w.vector_search_endpoints.delete_endpoint(VS_ENDPOINT_NAME)
    print(f"✓ Deleted VS endpoint: {VS_ENDPOINT_NAME} — billing stopped!")
except Exception as e:
    print(f"  VS endpoint '{VS_ENDPOINT_NAME}' not found: {e}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 5: Delete UC Function

# COMMAND ----------
try:
    spark.sql(f"DROP FUNCTION IF EXISTS {UC_TOOL_FUNCTION}")
    print(f"✓ Dropped UC function: {UC_TOOL_FUNCTION}")
except Exception as e:
    print(f"  UC function not found: {e}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 6: Delete Delta Tables

# COMMAND ----------
tables_to_delete = [
    TABLE_CHUNKS,
    TABLE_EMBEDDINGS,
    TABLE_GENIE,
    f"{CATALOG}.{SCHEMA}.eval_results",
]

for table in tables_to_delete:
    try:
        spark.sql(f"DROP TABLE IF EXISTS {table}")
        print(f"✓ Dropped table: {table}")
    except Exception as e:
        print(f"  Table '{table}' not found: {e}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 7: Delete Volume Contents (optional)

# COMMAND ----------
DELETE_VOLUME = False  # ← Set to True to also delete the Volume

if DELETE_VOLUME:
    try:
        spark.sql(f"DROP VOLUME IF EXISTS {CATALOG}.{SCHEMA}.{VOLUME_NAME}")
        print(f"✓ Dropped volume: {CATALOG}.{SCHEMA}.{VOLUME_NAME}")
    except Exception as e:
        print(f"  Volume not found: {e}")
else:
    print("  Volume kept (set DELETE_VOLUME = True to delete)")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 8: Delete Schema (optional — only if empty)

# COMMAND ----------
DELETE_SCHEMA = False  # ← Set to True to also delete the schema

if DELETE_SCHEMA:
    try:
        spark.sql(f"DROP SCHEMA IF EXISTS {CATALOG}.{SCHEMA} CASCADE")
        print(f"✓ Dropped schema: {CATALOG}.{SCHEMA}")
    except Exception as e:
        print(f"  Schema not found: {e}")
else:
    print(f"  Schema kept: {CATALOG}.{SCHEMA}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------
print("""
╔═══════════════════════════════════════════════════════╗
║              CLEANUP COMPLETE                         ║
╠═══════════════════════════════════════════════════════╣
║                                                       ║
║  Deleted:                                             ║
║    ✓ Supervisor agent                                 ║
║    ✓ Serving endpoint (agent + review)                ║
║    ✓ Vector Search index                              ║
║    ✓ Vector Search endpoint (24/7 billing stopped!)   ║
║    ✓ UC function                                      ║
║    ✓ Delta tables                                     ║
║                                                       ║
║  Kept (change flags above to delete):                 ║
║    • UC Volume (raw documents)                        ║
║    • Schema + Catalog                                 ║
║    • MLflow experiment + logged models                ║
║    • Genie Space (delete via UI)                      ║
║                                                       ║
║  To re-deploy: run notebooks 01 → 07 again.           ║
║  Models are still registered in UC — just redeploy.   ║
║                                                       ║
╚═══════════════════════════════════════════════════════╝
""")
