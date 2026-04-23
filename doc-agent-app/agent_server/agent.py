"""
LangGraph doc search agent — Databricks Apps variant (async).

Same agent logic as ../03_agent.py (Model Serving), repackaged for Apps:

  Model Serving (03_agent.py)          Apps (this file)
  ─────────────────────────            ─────────────────
  class LangGraphDocAgent              @invoke() / @stream() functions
    (ResponsesAgent)                   (async, no class needed)
  mlflow.models.ModelConfig            os.environ / .env
  graph.stream() (sync)                graph.astream() (async)
  manual event construction            process_agent_astream_events (utils.py)
  mlflow.models.set_model(AGENT)       (not needed — decorators auto-register)

Started via: uv run python start_server.py --port 8181
"""

import os
from typing import AsyncGenerator

import mlflow
import mlflow.deployments
from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    to_chat_completions_input,
)

from .utils import process_agent_astream_events

load_dotenv()

# --- Configuration via environment variables ---
# .env for local dev, app.yaml env: for deployed app.
VS_INDEX = os.environ.get("VS_INDEX", "my_catalog.agent_lab.docs_index")
LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT", "databricks-gpt-oss-120b")
EMBEDDING_ENDPOINT = os.environ.get("EMBEDDING_ENDPOINT", "databricks-gte-large-en")
SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    (
        "You are a helpful assistant that answers questions about "
        "LangChain documentation using a vector search index. "
        "Always cite your sources when using retrieved documents. "
        "If you don't know the answer, say so honestly."
    ),
)

# --- Tools ---
_w = WorkspaceClient(profile=os.environ.get("DATABRICKS_CONFIG_PROFILE", "DEFAULT"))
_vs_client = _w.vector_search_indexes
_deploy_client = mlflow.deployments.get_deploy_client("databricks")


@tool
def search_docs(query: str) -> str:
    """Search LangChain documentation for relevant information.

    Use this tool when the user asks questions about LangChain concepts,
    APIs, patterns, or best practices.
    """
    return "10 docs"
    # return _search_docs_impl(query)


@mlflow.trace(span_type="RETRIEVER", name="search_docs_retrieval")
def _search_docs_impl(query: str) -> str:
    """Inner retrieval logic — labeled as RETRIEVER for MLflow scorers."""
    resp = _deploy_client.predict(
        endpoint=EMBEDDING_ENDPOINT,
        inputs={"input": [query]},
    )
    query_vector = resp.data[0]["embedding"]

    results = _vs_client.query_index(
        index_name=VS_INDEX,
        columns=["content", "source"],
        query_vector=query_vector,
        num_results=3,
    )

    if not results.result.data_array:
        return "No relevant documents found."
    return "\n\n---\n\n".join(f"Source: {row[1]}\n{row[0]}" for row in results.result.data_array)


ALL_TOOLS = [search_docs]

# --- Agent ---
# NOTE: mlflow.langchain.autolog() disabled — it buffers LLM streaming,
# collapsing token-level deltas into a single chunk.
# Re-enable once mlflow fixes async streaming compatibility.
llm = ChatDatabricks(endpoint=LLM_ENDPOINT)


# --- Entry points ---
@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    """Non-streaming — collects all stream events into a single response."""
    outputs = [
        event.item
        async for event in stream_handler(request)
        if event.type == "response.output_item.done"
    ]
    return ResponsesAgentResponse(output=outputs)


@stream()
async def stream_handler(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    """Streaming — yields Responses API events from the LangGraph ReAct loop.

    stream_mode=["updates", "messages"]:
      "updates" → function_call, function_call_output, message done items
      "messages" → text deltas (token-by-token for streaming-capable models)
    """
    graph = create_agent(model=llm, tools=ALL_TOOLS, system_prompt=SYSTEM_PROMPT)
    chat_input = to_chat_completions_input([i.model_dump() for i in request.input])

    async for event in process_agent_astream_events(
        graph.astream(
            input={"messages": chat_input},
            stream_mode=["updates", "messages"],
        )
    ):
        yield event
