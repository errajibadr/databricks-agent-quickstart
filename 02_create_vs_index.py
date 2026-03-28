# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # 02 — Create Vector Search Index
# MAGIC
# MAGIC **Creates:** Embeddings table → VS Endpoint → Self-managed Delta Sync Index
# MAGIC
# MAGIC **Time:** ~15 minutes (endpoint creation + embedding + sync)
# MAGIC **Cost:** VS endpoint bills 24/7 once created — **delete with 99_cleanup when done!**
# MAGIC
# MAGIC ```
# MAGIC ┌──────────────────────────────────────────────────────────────┐
# MAGIC │                                                              │
# MAGIC │  docs_chunked (from notebook 01)                            │
# MAGIC │       │                                                      │
# MAGIC │       ▼  ai_query('databricks-gte-large-en', content)       │
# MAGIC │                                                              │
# MAGIC │  docs_with_embeddings (Delta table + 1024-dim vectors)      │
# MAGIC │       │                                                      │
# MAGIC │       ▼  Delta Sync (TRIGGERED)                             │
# MAGIC │                                                              │
# MAGIC │  docs_index (self-managed, on vs-endpoint-lab)              │
# MAGIC │       │                                                      │
# MAGIC │       ▼  query_vector required (you embed the query)        │
# MAGIC │                                                              │
# MAGIC └──────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC **Why self-managed?** You control the embedding pipeline (faster batch processing
# MAGIC via `ai_query()`). Managed embeddings are ~1 row/s on pay-per-token — too slow
# MAGIC for 100+ chunks. Tradeoff: you must embed queries yourself at query time.

# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1: Chunk Documents (only if using Option B from notebook 01)
# MAGIC
# MAGIC If you used Option A (bundled sample), `docs_chunked` already exists — skip to Step 2.
# MAGIC If you used Option B (live download to Volume), run this cell to chunk the raw markdown.

# COMMAND ----------
# Check if chunks table already exists
try:
    count = spark.sql(f"SELECT COUNT(*) as cnt FROM {TABLE_CHUNKS}").collect()[0]["cnt"]
    print(f"✓ {TABLE_CHUNKS} already exists with {count} chunks — skipping chunking")
    NEEDS_CHUNKING = False
except Exception:
    print(f"  {TABLE_CHUNKS} not found — will chunk from Volume files")
    NEEDS_CHUNKING = True

# COMMAND ----------
import hashlib
from pyspark.sql.functions import col, udf, explode
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, IntegerType

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

if NEEDS_CHUNKING:
    chunk_schema = ArrayType(StructType([
        StructField("id", StringType(), False),
        StructField("content", StringType(), False),
        StructField("source", StringType(), False),
        StructField("chunk_index", IntegerType(), False),
    ]))

    @udf(chunk_schema)
    def chunk_document(path: str, content_bytes: bytes):
        source = path.split("/")[-1]
        content = content_bytes.decode("utf-8", errors="replace")
        paragraphs = content.split("\n\n")
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current_chunk) + len(para) > CHUNK_SIZE and current_chunk:
                chunk_id = hashlib.md5(f"{source}:{len(chunks)}:{current_chunk[:50]}".encode()).hexdigest()
                chunks.append((chunk_id, current_chunk.strip(), source, len(chunks)))
                overlap_text = current_chunk[-CHUNK_OVERLAP:] if len(current_chunk) > CHUNK_OVERLAP else current_chunk
                current_chunk = overlap_text + "\n\n" + para
            else:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para

        if current_chunk.strip():
            chunk_id = hashlib.md5(f"{source}:{len(chunks)}:{current_chunk[:50]}".encode()).hexdigest()
            chunks.append((chunk_id, current_chunk.strip(), source, len(chunks)))
        return chunks

    raw_df = spark.read.format("binaryFile").load(f"{VOLUME_PATH}/langchain-docs/*.md")
    chunks_df = raw_df.select(
        explode(chunk_document(col("path"), col("content"))).alias("chunk")
    ).select("chunk.id", "chunk.content", "chunk.source", "chunk.chunk_index")

    chunks_df.write.format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .option("delta.enableChangeDataFeed", "true") \
        .saveAsTable(TABLE_CHUNKS)

    print(f"✓ Chunked and saved {chunks_df.count()} chunks to {TABLE_CHUNKS}")
else:
    print("Chunking skipped — table already exists")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2: Compute Embeddings with ai_query()
# MAGIC
# MAGIC `ai_query()` is a SQL function that calls any Foundation Model API endpoint.
# MAGIC We use it in batch to embed all chunks — much faster than Delta Sync's
# MAGIC sequential approach (~1 row/s on pay-per-token).
# MAGIC
# MAGIC **GOTCHA:** `ai_query()` sends the entire batch as one HTTP request.
# MAGIC If the combined text exceeds 4MB, it fails. Batch size of 50 works well.

# COMMAND ----------
# Compute embeddings in batches via ai_query()
BATCH_SIZE = 50

total_rows = spark.sql(f"SELECT COUNT(*) as cnt FROM {TABLE_CHUNKS}").collect()[0]["cnt"]
print(f"Embedding {total_rows} chunks in batches of {BATCH_SIZE}...")

for offset in range(0, total_rows, BATCH_SIZE):
    if offset == 0:
        # First batch: CREATE OR REPLACE to reset CDF checkpoint
        spark.sql(f"""
            CREATE OR REPLACE TABLE {TABLE_EMBEDDINGS}
            TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
            AS SELECT
                id, content, source, chunk_index,
                ai_query('{EMBEDDING_ENDPOINT}', content) AS embedding
            FROM (
                SELECT *, ROW_NUMBER() OVER (ORDER BY id) as rn
                FROM {TABLE_CHUNKS}
            )
            WHERE rn > {offset} AND rn <= {offset + BATCH_SIZE}
        """)
    else:
        # Subsequent batches: append
        spark.sql(f"""
            INSERT INTO {TABLE_EMBEDDINGS}
            SELECT
                id, content, source, chunk_index,
                ai_query('{EMBEDDING_ENDPOINT}', content) AS embedding
            FROM (
                SELECT *, ROW_NUMBER() OVER (ORDER BY id) as rn
                FROM {TABLE_CHUNKS}
            )
            WHERE rn > {offset} AND rn <= {offset + BATCH_SIZE}
        """)
    batch_num = offset // BATCH_SIZE + 1
    total_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  Batch {batch_num}/{total_batches}: rows {offset + 1}-{min(offset + BATCH_SIZE, total_rows)}")

embedded_count = spark.sql(f"SELECT COUNT(*) as cnt FROM {TABLE_EMBEDDINGS}").collect()[0]["cnt"]
print(f"\n✓ Embeddings complete: {embedded_count} rows in {TABLE_EMBEDDINGS}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3: Create Vector Search Endpoint
# MAGIC
# MAGIC The endpoint is the **compute** that serves similarity queries.
# MAGIC It bills 24/7 — no scale-to-zero. Delete with `99_cleanup` when done!

# COMMAND ----------
from databricks.sdk.service.vectorsearch import EndpointType
import time

# Create VS endpoint (idempotent)
try:
    existing = _w.vector_search_endpoints.get_endpoint(VS_ENDPOINT_NAME)
    print(f"✓ Endpoint exists: {existing.name} (status: {existing.endpoint_status})")
except Exception:
    print(f"Creating endpoint '{VS_ENDPOINT_NAME}' (takes 5-10 min)...")
    _w.vector_search_endpoints.create_endpoint(
        name=VS_ENDPOINT_NAME,
        endpoint_type=EndpointType.STANDARD,
    )

# Wait for ONLINE status
for i in range(30):
    ep = _w.vector_search_endpoints.get_endpoint(VS_ENDPOINT_NAME)
    state = ep.endpoint_status.state.value if ep.endpoint_status and ep.endpoint_status.state else "UNKNOWN"
    print(f"  [{i}] Status: {state}")
    if state == "ONLINE":
        print("✓ Endpoint is ONLINE!")
        break
    time.sleep(30)
else:
    print("⚠ Endpoint not ready after 15 min — check Compute > Vector Search in the UI")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4: Create Self-Managed Delta Sync Index
# MAGIC
# MAGIC Self-managed = you pre-computed embeddings. Delta Sync just indexes the vectors.
# MAGIC Fast sync, but `query_text` won't work — you must pass `query_vector`.

# COMMAND ----------
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingVectorColumn,
    VectorIndexType,
    PipelineType,
)

try:
    existing = _w.vector_search_indexes.get_index(VS_INDEX_NAME)
    print(f"✓ Index exists: {existing.name} (status: {existing.status})")
except Exception:
    print(f"Creating index '{VS_INDEX_NAME}'...")
    _w.vector_search_indexes.create_index(
        name=VS_INDEX_NAME,
        endpoint_name=VS_ENDPOINT_NAME,
        primary_key="id",
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=TABLE_EMBEDDINGS,
            embedding_vector_columns=[
                EmbeddingVectorColumn(name="embedding", embedding_dimension=1024),
            ],
            pipeline_type=PipelineType.TRIGGERED,
            columns_to_sync=["content", "source", "chunk_index"],
        ),
    )
    print("✓ Index created — triggering sync...")

# COMMAND ----------
# Trigger sync and wait
_w.vector_search_indexes.sync_index(index_name=VS_INDEX_NAME)

for i in range(20):
    idx = _w.vector_search_indexes.get_index(VS_INDEX_NAME)
    status = idx.status
    print(f"  [{i}] Index status: {status}")
    if status and status.ready:
        print("✓ Index is ONLINE and synced!")
        break
    time.sleep(30)
else:
    print("⚠ Not ready after 10 min — check Vector Search in Catalog Explorer")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 5: Test Query

# COMMAND ----------
import mlflow.deployments

deploy_client = mlflow.deployments.get_deploy_client("databricks")

query = "What is tool calling in LangChain?"

# Embed the query (self-managed = you embed)
response = deploy_client.predict(
    endpoint=EMBEDDING_ENDPOINT,
    inputs={"input": [query]},
)
query_vector = response.data[0]["embedding"]

# Query the index
results = _w.vector_search_indexes.query_index(
    index_name=VS_INDEX_NAME,
    query_vector=query_vector,
    columns=["content", "source"],
    num_results=3,
)

print(f"Query: {query}\n")
for row in results.result.data_array:
    score = row[-1]
    print(f"  [{score:.4f}] {row[1]}")
    print(f"           {row[0][:120]}...\n")

print("✓ Vector Search is working! Proceed to notebook 03/04 (agent).")
