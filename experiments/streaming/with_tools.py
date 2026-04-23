"""Test: does create_agent stream differently with tools vs without?"""

import os
import time
from dotenv import load_dotenv

load_dotenv()

from databricks_langchain import ChatDatabricks
from langchain.agents import create_agent
from langchain_core.tools import tool

LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT", "databricks-gpt-oss-120b")


@tool
def dummy_tool(query: str) -> str:
    """A dummy tool that does nothing."""
    return "result"


llm = ChatDatabricks(endpoint=LLM_ENDPOINT)


async def test(tools, label):
    print(f"\n{'=' * 60}")
    print(f"TEST: create_agent with {label}")
    print(f"{'=' * 60}")
    graph = create_agent(model=llm, tools=tools, system_prompt="You are helpful.")
    messages = {"messages": [{"role": "user", "content": "tell me a joke"}]}
    start = time.time()
    count = 0
    content_count = 0
    async for part in graph.astream(input=messages, stream_mode="messages", version="v2"):
        if part["type"] == "messages":
            chunk, meta = part["data"]
            count += 1
            has = bool(getattr(chunk, "content", None))
            if has:
                content_count += 1
            elapsed = f"{time.time() - start:.2f}s"
            print(f"  [{elapsed}] #{count} has_content={has} type={type(chunk).__name__}")
    print(f"\n  Total: {count} events, {content_count} with content")


async def test_raw_bind_tools():
    """Raw llm.bind_tools().astream() — no LangGraph at all."""
    print(f"\n{'=' * 60}")
    print("TEST: llm.bind_tools([dummy_tool]).astream() — NO LangGraph")
    print(f"{'=' * 60}")
    llm_with_tools = llm.bind_tools([dummy_tool])
    start = time.time()
    count = 0
    content_count = 0
    async for chunk in llm_with_tools.astream([{"role": "user", "content": "tell me a joke"}]):
        count += 1
        has = bool(getattr(chunk, "content", None))
        if has:
            content_count += 1
        elapsed = f"{time.time() - start:.2f}s"
        print(f"  [{elapsed}] #{count} has_content={has}")
    print(f"\n  Total: {count} events, {content_count} with content")


# asyncio.run(test([], "tools=[]"))
# asyncio.run(test([dummy_tool], "tools=[dummy_tool]"))
# asyncio.run(test_raw_bind_tools())
from databricks_langchain import ChatDatabricks


@tool
def dummy_tool(query: str) -> str:
    """A dummy tool that does nothing."""
    return "result"


llm = ChatDatabricks(
    endpoint="databricks-llama-4-maverick",  # "databricks-qwen3-next-80b-a3b-instruct",  # or your serving endpoint name
    temperature=0.0,
    # use_responses_api=True,
    # streaming=True,
    # callbacks=[],
)
# llm = llm.bind_tools([dummy_tool])

for chunk in llm.stream("Write a 1-sentence summary of Databricks.", stream_usage=True):
    print(repr(chunk))
