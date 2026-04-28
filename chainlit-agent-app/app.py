"""Chainlit handler — wires backend → normalizer → renderer.

Architecture (mirrors `_references/bi-hub-app/src/app/routes.py`):

    @cl.on_message
        backend.stream(history)         # raw Responses-API events
            │
            ▼
        services.normalize(...)         # → message.start / text.delta /
            │                             tool.call / tool.output / thought
            ▼
        ChainlitStream                  # one cl.Message per output item,
                                          chronologically ordered (Path B)

See annex §17 of `creative_phase_2026-04-27_dbx_apps_streaming_agents.md`
for why per-item bubbles replaced the bi-hub status-aggregator pattern.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Loaded at import time so 03_agent.py can read DATABRICKS_HOST / VS_INDEX
# during its module-level WorkspaceClient() construction.
load_dotenv()

import chainlit as cl

# Side-effect import: registers `@cl.header_auth_callback`. Active only when
# CHAINLIT_AUTH_SECRET is set (i.e. Apps deployment); a no-op in local dev.
import auth  # noqa: F401

from backends.base import Backend
from backends.endpoint import EndpointBackend
from backends.local_agent import LocalAgentBackend
from services import ChainlitStream, normalize


def _obo_token_from_session() -> str | None:
    """Pull the OBO token off the authenticated `cl.User`, if present.

    `auth.py` stashes it on `cl.User.metadata["obo_token"]` during the
    header-auth callback. In local dev (no auth) `session.user` is None and
    we return None — the EndpointBackend then falls back to the standard
    WorkspaceClient auth chain.
    """
    session = getattr(cl.context, "session", None)
    user = getattr(session, "user", None) if session else None
    metadata = getattr(user, "metadata", None) or {}
    return metadata.get("obo_token")


def _build_backend() -> Backend:
    backend_type = os.environ.get("BACKEND", "local").lower()

    if backend_type == "local":
        default_module = str(Path(__file__).resolve().parent.parent / "03_agent.py")
        return LocalAgentBackend(
            module_path=os.environ.get("LOCAL_AGENT_MODULE", default_module)
        )

    if backend_type == "endpoint":
        # Single construction site for both deployment lanes ("Option A" in the
        # design doc). Trade-off: couples this factory to Chainlit's request
        # internals — acceptable v1, revisit if a non-Chainlit caller ever
        # needs `EndpointBackend` (would split header-read into a wrapper).
        #
        # Lane L2 (local): obo_token is None → WorkspaceClient walks the
        # standard auth chain (DEFAULT profile / .env / SP).
        # Lane L3 (Apps OBO): obo_token is the user's bearer from
        # `x-forwarded-access-token`, captured by `auth.py`.
        return EndpointBackend.from_env(obo_token=_obo_token_from_session())

    raise ValueError(f"Unknown BACKEND={backend_type!r}. Use 'local' or 'endpoint'.")


@cl.on_chat_start
async def on_chat_start():
    try:
        backend = _build_backend()
    except Exception as exc:
        # Surface backend-construction errors at session start instead of
        # silently leaving `backend=None` for `on_message` to crash on.
        await cl.Message(
            content=(
                f"**Failed to initialize backend** "
                f"(`{type(exc).__name__}: {exc}`). "
                f"Check logs / env vars and reload the page."
            ),
            author="system",
        ).send()
        raise
    cl.user_session.set("backend", backend)
    cl.user_session.set("history", [])


@cl.on_message
async def on_message(message: cl.Message):
    backend: Backend | None = cl.user_session.get("backend")
    if backend is None:
        await cl.Message(
            content=(
                "Backend was not initialized — see the error at the top of "
                "this chat or reload the page."
            ),
            author="system",
        ).send()
        return
    history: list[dict] = cl.user_session.get("history") or []
    history.append({"role": "user", "content": message.content})

    renderer = ChainlitStream()
    final_text_parts: list[str] = []

    try:
        async for evt in normalize(backend.stream(history)):
            kind = evt["type"]
            if kind == "message.start":
                await renderer.on_message_start(evt["item_id"])
            elif kind == "text.delta":
                token = evt["delta"]
                final_text_parts.append(token)
                await renderer.on_text_delta(evt.get("item_id", ""), token)
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
