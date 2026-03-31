"""
LangGraph doc search agent — Databricks Apps variant.

Same agent logic as ../03_agent.py (Model Serving), repackaged for Apps:

  Model Serving (03_agent.py)          Apps (this file)
  ─────────────────────────            ─────────────────
  class LangGraphDocAgent              @invoke() / @stream() functions
    (ResponsesAgent)                   (no class needed)
  mlflow.models.ModelConfig            os.environ / .env
  mlflow.models.set_model(AGENT)       (not needed — decorators auto-register)
  log_model() → agents.deploy()        databricks bundle deploy + run

Started via: mlflow genai serve --module agent_server.agent
"""

import os
import mlflow
import mlflow.deployments
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    to_chat_completions_input,
)
from dotenv import load_dotenv

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


# --- Tools (same logic as 03_agent.py) ---
from databricks.sdk import WorkspaceClient
from langchain_core.tools import tool

print("DATABRICKS_CONFIG_PROFILE", os.environ.get("DATABRICKS_CONFIG_PROFILE"))
_w = WorkspaceClient(profile=os.environ.get("DATABRICKS_CONFIG_PROFILE"))
_vs_client = _w.vector_search_indexes
_deploy_client = mlflow.deployments.get_deploy_client("databricks")


@tool
def search_docs(query: str) -> str:
    """Search LangChain documentation for relevant information.

    Use this tool when the user asks questions about LangChain concepts,
    APIs, patterns, or best practices.
    """
    return _search_docs_impl(query)


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


# --- LangGraph setup (module-level — no class wrapper) ---
from databricks_langchain import ChatDatabricks
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode
from typing import Annotated, Sequence, TypedDict

llm = ChatDatabricks(endpoint=LLM_ENDPOINT)
llm_with_tools = llm.bind_tools(ALL_TOOLS)


class AgentState(TypedDict):
    messages: Annotated[Sequence, add_messages]


def _build_graph():
    def should_continue(state):
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "end"

    def call_model(state):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(state["messages"])
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", RunnableLambda(call_model))
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    graph.set_entry_point("agent")
    return graph.compile()


# --- Entry points ---
mlflow.langchain.autolog()


@invoke()
def non_streaming(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    """Non-streaming — collects all stream events into a single response."""
    outputs = [event.item for event in streaming(request) if event.type == "response.output_item.done"]
    return ResponsesAgentResponse(output=outputs)


@stream()
def streaming(request: ResponsesAgentRequest):
    """Streaming — yields ResponsesAgentStreamEvents as they arrive."""
    messages = to_chat_completions_input([m.model_dump() for m in request.input])
    graph = _build_graph()
    msg_id = "msg_1"
    text_parts = []

    for msg, metadata in graph.stream({"messages": messages}, stream_mode="messages"):
        if isinstance(msg, AIMessageChunk):
            if msg.tool_call_chunks:
                for tc in msg.tool_call_chunks:
                    if tc.get("name"):
                        yield ResponsesAgentStreamEvent(
                            type="response.output_item.done",
                            item={
                                "type": "function_call",
                                "id": tc.get("id", ""),
                                "call_id": tc.get("id", ""),
                                "name": tc["name"],
                                "arguments": tc.get("args", "{}"),
                            },
                        )
            elif msg.content:
                text_parts.append(msg.content)
                yield ResponsesAgentStreamEvent(
                    type="response.content_part.delta",
                    item_id=msg_id,
                    content_index=0,
                    delta=msg.content,
                )
        elif isinstance(msg, ToolMessage):
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item={
                    "type": "function_call_output",
                    "call_id": msg.tool_call_id,
                    "output": str(msg.content),
                },
            )

    if text_parts:
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item={
                "type": "message",
                "id": msg_id,
                "role": "assistant",
                "content": [{"type": "output_text", "text": "".join(text_parts)}],
            },
        )
