"""
Standalone LangGraph agent for the Workspace Kit.

This file is what log_model(python_model="03_agent.py") packages and what the
serving container re-executes. It must be self-contained:
  - All imports
  - Config via model_config (externalizable settings)
  - Tool definitions
  - Agent class (ResponsesAgent subclass)
  - AGENT = ... and mlflow.models.set_model(AGENT) at module level

Architecture:
    ┌──────────────────────────────────────────┐
    │        ResponsesAgent wrapper             │
    │  (translates HTTP ↔ LangGraph state)     │
    └──────────────────┬───────────────────────┘
                       │
         ┌─────────────▼─────────────────────┐
         │       LangGraph ReAct Loop         │
         │                                     │
         │  ┌───────┐  tool_calls?  ┌───────┐ │
         │  │ Agent │ ── YES ─────► │ Tools │ │
         │  │ (LLM) │ ◄── results ─ │ Node  │ │
         │  │       │ ── NO ──► END └───────┘ │
         │  └───────┘                          │
         └─────────────────────────────────────┘
                       │
         Tool: search_docs (Vector Search — self-managed embeddings)
"""

import mlflow
import mlflow.deployments
from mlflow.pyfunc.model import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    to_chat_completions_input,
)

# --- Configuration via environment variables ---
# Set at deploy time via agents.deploy(environment_vars={...}), or locally via `export`.
# LLM and embedding endpoints default to Databricks system endpoints available in every
# workspace. VS_INDEX is workspace-specific — we fail loud if missing rather than guess.
import os

VS_INDEX = os.environ.get("VS_INDEX")
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

if not VS_INDEX:
    raise RuntimeError(
        "VS_INDEX env var is required. "
        "Local dev: `export VS_INDEX=your_catalog.your_schema.docs_index`. "
        "Deployment: pass via agents.deploy(environment_vars={'VS_INDEX': VS_INDEX_NAME, ...})."
    )


# --- Tools ---
from databricks.sdk import WorkspaceClient
from langchain_core.tools import tool

_w = WorkspaceClient()
_vs_client = _w.vector_search_indexes
_deploy_client = mlflow.deployments.get_deploy_client("databricks")


@tool
def search_docs(query: str) -> str:
    """Search LangChain documentation for relevant information.

    Use this tool when the user asks questions about LangChain concepts,
    APIs, patterns, or best practices.
    """
    return _search_docs_impl(query)


@tool
def meteo_forecast(city: str) -> str:
    """Get the weather forecast for a given city."""
    return f"The weather in {city} is sunny and 22°C."


@mlflow.trace(span_type="RETRIEVER", name="search_docs_retrieval")
def _search_docs_impl(query: str) -> str:
    """Inner retrieval logic — labeled as RETRIEVER for MLflow scorers."""
    # Embed query (self-managed index requires query_vector)
    resp = _deploy_client.predict(
        endpoint=EMBEDDING_ENDPOINT,
        inputs={"input": [query]},
    )
    query_vector = resp.data[0]["embedding"]

    # Query Vector Search
    results = _vs_client.query_index(
        index_name=VS_INDEX,
        columns=["content", "source"],
        query_vector=query_vector,
        num_results=3,
    )

    if not results.result.data_array:
        return "No relevant documents found."
    return "\n\n---\n\n".join(f"Source: {row[1]}\n{row[0]}" for row in results.result.data_array)


ALL_TOOLS = [search_docs, meteo_forecast]


# --- Agent ---
from databricks_langchain import ChatDatabricks
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing import Annotated, Generator, Sequence, TypedDict


class AgentState(TypedDict):
    messages: Annotated[Sequence, add_messages]


class LangGraphDocAgent(ResponsesAgent):
    """
    LangGraph ReAct agent for LangChain doc search, wrapped in ResponsesAgent.

    Simplified from Phase 3's 03_agent.py — single tool (search_docs),
    no UC functions. Designed to be wrapped as a UC tool for Supervisor.
    """

    def __init__(self):
        self.llm = ChatDatabricks(endpoint=LLM_ENDPOINT)
        self.tools = ALL_TOOLS
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def _build_graph(self):
        def should_continue(state):
            last = state["messages"][-1]
            if isinstance(last, AIMessage) and last.tool_calls:
                return "tools"
            return "end"

        def call_model(state):
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(state["messages"])
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}

        graph = StateGraph(AgentState)
        graph.add_node("agent", RunnableLambda(call_model))
        graph.add_node("tools", ToolNode(self.tools))
        graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
        graph.add_edge("tools", "agent")
        graph.set_entry_point("agent")
        return graph.compile()

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [event.item for event in self.predict_stream(request) if event.type == "response.output_item.done"]
        return ResponsesAgentResponse(output=outputs)

    def predict_stream(self, request: ResponsesAgentRequest) -> Generator[ResponsesAgentStreamEvent, None, None]:
        messages = to_chat_completions_input([m.model_dump() for m in request.input])
        graph = self._build_graph()
        msg_id = "msg_1"
        text_parts = []

        for msg, metadata in graph.stream({"messages": messages}, stream_mode="messages"):
            if isinstance(msg, AIMessageChunk):
                if msg.tool_call_chunks:
                    for tc in msg.tool_call_chunks:
                        if tc.get("name"):
                            yield ResponsesAgentStreamEvent(
                                type="response.output_item.done",
                                item=self.create_function_call_item(
                                    id=tc.get("id", ""),
                                    call_id=tc.get("id", ""),
                                    name=tc["name"],
                                    arguments=tc.get("args", "{}"),
                                ),
                            )
                elif msg.content:
                    text_parts.append(msg.content)
                    yield ResponsesAgentStreamEvent(
                        **self.create_text_delta(delta=msg.content, item_id=msg_id),
                    )
            elif isinstance(msg, ToolMessage):
                yield ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item=self.create_function_call_output_item(
                        call_id=msg.tool_call_id,
                        output=str(msg.content),
                    ),
                )

        if text_parts:
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.create_text_output_item(text="".join(text_parts), id=msg_id),
            )


# --- MLflow registration ---
mlflow.langchain.autolog()
AGENT = LangGraphDocAgent()
mlflow.models.set_model(AGENT)
