"""Chainlit handler — wires backend → normalizer → renderer.

Architecture (mirrors `_references/bi-hub-app/src/app/routes.py`):

    @cl.on_message
        backend.stream(history)         # raw Responses-API events
            │
            ▼
        services.normalize(...)         # → text.delta / tool.call /
            │                             tool.output / thought
            ▼
        ChainlitStream                  # status_msg (aggregator) + text_msg (response)

Step A scope: Lane L1 only (LocalAgentBackend). EndpointBackend lands in Step B.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Loaded at import time so 03_agent.py can read DATABRICKS_HOST / VS_INDEX
# during its module-level WorkspaceClient() construction.
load_dotenv()

import chainlit as cl

from backends.base import Backend
from backends.local_agent import LocalAgentBackend
from services import ChainlitStream, normalize


def _build_backend() -> Backend:
    backend_type = os.environ.get("BACKEND", "local").lower()

    if backend_type == "local":
        default_module = str(Path(__file__).resolve().parent.parent / "03_agent.py")
        return LocalAgentBackend(
            module_path=os.environ.get("LOCAL_AGENT_MODULE", default_module)
        )

    if backend_type == "endpoint":
        raise NotImplementedError(
            "EndpointBackend lands in Step B. Set BACKEND=local for now."
        )

    raise ValueError(f"Unknown BACKEND={backend_type!r}. Use 'local' or 'endpoint'.")


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("backend", _build_backend())
    cl.user_session.set("history", [])


@cl.on_message
async def on_message(message: cl.Message):
    backend: Backend = cl.user_session.get("backend")
    history: list[dict] = cl.user_session.get("history") or []
    history.append({"role": "user", "content": message.content})

    renderer = ChainlitStream()
    final_text_parts: list[str] = []

    try:
        async for evt in normalize(backend.stream(history)):
            kind = evt["type"]
            if kind == "text.delta":
                token = evt["delta"]
                final_text_parts.append(token)
                await renderer.on_text_delta(token)
            elif kind == "thought":
                await renderer.on_thought(evt["text"])
            elif kind == "tool.call":
                await renderer.on_tool_call(evt["call_id"], evt["name"], evt["args"])
            elif kind == "tool.output":
                await renderer.on_tool_output(evt["call_id"], evt["output"])
    except Exception as exc:
        await cl.Message(
            content=f"Backend error: `{type(exc).__name__}: {exc}`"
        ).send()
        raise
    finally:
        await renderer.finalize()

    history.append({"role": "assistant", "content": "".join(final_text_parts)})
    cl.user_session.set("history", history)
