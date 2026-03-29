# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # 01 — Setup Foundation
# MAGIC
# MAGIC **Creates:** Catalog → Schema → Volume → Uploads sample documents
# MAGIC
# MAGIC **Time:** ~2 minutes | **Cost:** Free (metadata operations only)
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────┐
# MAGIC │  Unity Catalog Hierarchy                            │
# MAGIC │                                                     │
# MAGIC │  my_catalog          ◄── Catalog           │
# MAGIC │    └── agent_lab              ◄── Schema            │
# MAGIC │          ├── documents/       ◄── Volume (raw docs) │
# MAGIC │          ├── docs_chunked     ◄── (created in 02)   │
# MAGIC │          └── docs_index       ◄── (created in 02)   │
# MAGIC └─────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC **Two data paths:**
# MAGIC - **Fast path:** Load bundled `sample_docs.json` (~100 chunks, instant, no internet)
# MAGIC - **Full path:** Download 741 docs from `llms.txt` (~2 min, needs internet)

# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1: Create Catalog, Schema, and Volume

# COMMAND ----------
# Create UC hierarchy — idempotent (IF NOT EXISTS)
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME_NAME}")

print(f"✓ Catalog:  {CATALOG}")
print(f"✓ Schema:   {CATALOG}.{SCHEMA}")
print(f"✓ Volume:   {VOLUME_PATH}")

# Verify
spark.sql(f"SHOW SCHEMAS IN {CATALOG} LIKE 'agent_lab'").show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2: Upload Documents
# MAGIC
# MAGIC Choose ONE of the two options below:
# MAGIC - **Option A** (recommended): Bundled sample — fast, deterministic, no internet
# MAGIC - **Option B**: Live download — full corpus, needs internet access from workspace

# COMMAND ----------
# MAGIC %md
# MAGIC ### Option A: Load Bundled Sample (recommended for first run)
# MAGIC
# MAGIC Loads ~100 pre-chunked docs from `data/sample_docs.json` included in the repo.
# MAGIC This skips the download + chunking steps entirely — chunks go straight to a Delta table.

# COMMAND ----------
import json
import os

# Detect the path to sample_docs.json relative to this notebook
# In Databricks Repos, __file__ doesn't exist — use the notebook context
try:
    _notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    # In Repos: /Repos/user/repo-name/agents/workspace_kit/01_setup_foundation
    # The data/ folder is a sibling directory
    _repo_root = "/Workspace" + "/".join(_notebook_path.split("/")[:-1])
    SAMPLE_DATA_PATH = f"{_repo_root}/data/sample_docs.json"
except Exception:
    SAMPLE_DATA_PATH = None

USE_BUNDLED_SAMPLE = True  # ← Set to False to use Option B (live download)

if USE_BUNDLED_SAMPLE and SAMPLE_DATA_PATH:
    # Read the bundled JSON file
    with open(SAMPLE_DATA_PATH, "r") as f:
        sample_chunks = json.load(f)

    print(f"Loaded {len(sample_chunks)} chunks from bundled sample")
    print(f"Sources: {sorted(set(c['source'] for c in sample_chunks))}")

    # Write directly to Delta table (skip Volume upload + chunking)
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType

    schema = StructType([
        StructField("id", StringType(), False),
        StructField("content", StringType(), False),
        StructField("source", StringType(), False),
        StructField("chunk_index", IntegerType(), False),
    ])

    df = spark.createDataFrame(sample_chunks, schema=schema)
    df.write.format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .option("delta.enableChangeDataFeed", "true") \
        .saveAsTable(TABLE_CHUNKS)

    print(f"✓ Saved {df.count()} chunks to {TABLE_CHUNKS}")
    print("  → Skip to notebook 02 (VS index creation)")
else:
    print("Bundled sample not used — run Option B below")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Option B: Download from llms.txt (full corpus)
# MAGIC
# MAGIC Downloads all LangChain docs from the llms.txt manifest.
# MAGIC Saves raw markdown to the UC Volume, then chunks in notebook 02.
# MAGIC
# MAGIC **Requires:** Internet access from the workspace cluster.

# COMMAND ----------
# Option B: Live download (skip this cell if you used Option A)

DOWNLOAD_LIVE = False  # ← Set to True to download live docs

if DOWNLOAD_LIVE:
    import requests
    import re
    import time

    LLMS_TXT_URL = "https://docs.langchain.com/llms.txt"

    print(f"Fetching {LLMS_TXT_URL}...")
    resp = requests.get(LLMS_TXT_URL, timeout=30)
    resp.raise_for_status()

    # Parse URLs from llms.txt
    entries = []
    for line in resp.text.splitlines():
        match = re.match(r"^- \[(.+?)\]\((.+?)\)", line.strip())
        if match:
            title, url = match.group(1), match.group(2)
            if not url.endswith(".json"):
                entries.append((title, url))

    print(f"Found {len(entries)} doc pages to download")

    # Download each page and save to Volume
    success, failed = 0, 0
    start = time.time()

    for i, (title, url) in enumerate(entries, 1):
        try:
            page_resp = requests.get(url, timeout=15)
            page_resp.raise_for_status()
            text = page_resp.text.strip()

            if len(text) < 20:
                continue

            # Create filename from URL path
            from urllib.parse import urlparse
            parsed = urlparse(url)
            name = parsed.path.strip("/").replace("/", "__")
            name = re.sub(r"\.md$", "", name)
            if len(name) > 120:
                name = name[:120]
            filename = f"{name}.md"

            # Write to Volume with frontmatter
            safe_title = title.replace('"', '\\"')
            content = f'---\ntitle: "{safe_title}"\nsource: "{url}"\n---\n\n{text}\n'

            dbutils.fs.put(
                f"{VOLUME_PATH}/langchain-docs/{filename}",
                content,
                overwrite=True,
            )
            success += 1

        except Exception as e:
            failed += 1

        if i % 50 == 0:
            elapsed = time.time() - start
            print(f"  [{i}/{len(entries)}] {success} saved, {failed} failed ({elapsed:.0f}s)")

    elapsed = time.time() - start
    print(f"\n✓ Done: {success} saved, {failed} failed ({elapsed:.0f}s)")
    print(f"  Files in: {VOLUME_PATH}/langchain-docs/")
    print()
    print("  ⚠ Option B does NOT create the docs_chunked table here.")
    print("    Notebook 02 will detect the missing table and chunk these")
    print("    Volume files into docs_chunked automatically before embedding.")
    print("    → Proceed to notebook 02.")
else:
    print("Live download disabled — set DOWNLOAD_LIVE = True to use this path")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3: Verify

# COMMAND ----------
# Verify: show what we have
print("=== Delta table (docs_chunked) ===")
try:
    count = spark.sql(f"SELECT COUNT(*) as cnt FROM {TABLE_CHUNKS}").collect()[0]["cnt"]
    print(f"  ✓ {TABLE_CHUNKS}: {count} chunks (Option A)")
    spark.sql(f"""
        SELECT source, COUNT(*) as chunks, ROUND(AVG(LENGTH(content))) as avg_len
        FROM {TABLE_CHUNKS}
        GROUP BY source ORDER BY chunks DESC LIMIT 10
    """).show(truncate=60)
except Exception:
    print(f"  Table not found — expected if you used Option B.")
    print(f"  Notebook 02 will chunk Volume files → {TABLE_CHUNKS} before embedding.")

print("\n=== Volume (raw files) ===")
try:
    files = dbutils.fs.ls(f"{VOLUME_PATH}/langchain-docs")
    print(f"  ✓ {len(files)} files in {VOLUME_PATH}/langchain-docs/ (Option B)")
except Exception:
    print(f"  No files in Volume (expected if using Option A — chunks are already in Delta)")

print("\n=== Next step ===")
print("  → Proceed to notebook 02 (Vector Search index creation)")
