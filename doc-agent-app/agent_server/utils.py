"""
Stream processing utilities for the Apps variant.

Converts LangGraph astream events into ResponsesAgentStreamEvent objects
that the MLflow GenAI Server can send as SSE to clients.

Two stream modes work together:
  "updates"  → complete messages after each node finishes
               (AIMessage with tool_calls, ToolMessage, final AIMessage)
               converted via output_to_responses_items_stream()
  "messages" → token-level text deltas (AIMessageChunk)
               converted via create_text_delta()

Tool calls arrive here via "updates" mode as complete items, not
token-by-token — same pattern as most providers for structured output.
"""

import json
import logging
from typing import Any, AsyncGenerator, AsyncIterator

from langchain_core.messages import AIMessageChunk, ToolMessage
from mlflow.types.responses import (
    ResponsesAgentStreamEvent,
    create_text_delta,
    output_to_responses_items_stream,
)

logger = logging.getLogger(__name__)


def _extract_text(content) -> str | None:
    """Extract displayable text from structured response content.

    Reasoning models (e.g. gpt-oss-120b) return content as structured blocks:
      [{"type": "reasoning", "summary": [...]}, {"type": "text", "text": "..."}]

    Content can arrive as:
      - Plain string: "Hello" → return as-is
      - JSON-serialized structured blocks: parse and extract text blocks only
      - List of dicts (raw, before serialization): same extraction
      - Empty / non-text: → None
    """
    if not content:
        return None
    if isinstance(content, str):
        if not content.startswith('[{"type":'):
            return content
        try:
            blocks = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            return content
        text_parts = [b["text"] for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
        return "".join(text_parts) if text_parts else None
    if isinstance(content, list):
        text_parts = [b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"]
        return "".join(text_parts) if text_parts else None
    return str(content)


async def process_agent_astream_events(
    async_stream: AsyncIterator[Any],
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    """Convert LangGraph astream(stream_mode=["updates", "messages"]) events
    into ResponsesAgentStreamEvent objects for the MLflow GenAI Server.
    """
    async for event in async_stream:
        mode, data = event[0], event[1]

        if mode == "updates":
            for node_data in data.values():
                messages = node_data.get("messages") or []
                for msg in messages:
                    if isinstance(msg, ToolMessage) and not isinstance(msg.content, str):
                        msg.content = json.dumps(msg.content)
                if messages:
                    for item in output_to_responses_items_stream(messages):
                        yield item

        elif mode == "messages":
            chunk, _metadata = data
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                text = _extract_text(chunk.content)
                if text:
                    yield ResponsesAgentStreamEvent(
                        **create_text_delta(delta=text, item_id=chunk.id)
                    )
