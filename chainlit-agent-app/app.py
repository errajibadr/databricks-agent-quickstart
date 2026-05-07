"""Chainlit handler — wires backend → normalizer → renderer.

Architecture (Chainlit handler → backend → normalizer → renderer):

    @cl.on_message
        backend.stream(history)         # raw Responses-API events
            │
            ▼
        services.normalize(...)         # → message.start / text.delta /
            │                             tool.call / tool.output / thought
            ▼
        ChainlitStream                  # one cl.Message per output item,
                                          chronologically ordered (Path B)

See chainlit-agent-app/README.md § "Design references" for the chronological renderer rationale.
for why per-item bubbles replaced the bi-hub status-aggregator pattern.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv


# Shares the "obo_diag" namespace with auth.py — handshake lines (auth) and
# turn lines (here) interleave in the App's log stream so the TTL investigation
# can correlate them. See
# memory_bank/creative_phases/creative_phase_2026-05-07_chainlit_obo_expiry.md.
logger = logging.getLogger("obo_diag")

# Loaded at import time so 03_agent.py can read DATABRICKS_HOST / VS_INDEX
# during its module-level WorkspaceClient() construction.
load_dotenv()

import chainlit as cl
import openai

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
        return LocalAgentBackend(module_path=os.environ.get("LOCAL_AGENT_MODULE", default_module))

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
            content=(f"**Failed to initialize backend** (`{type(exc).__name__}: {exc}`). Check logs / env vars and reload the page."),
            author="system",
        ).send()
        raise
    cl.user_session.set("backend", backend)
    cl.user_session.set("history", [])


def _obo_expired() -> bool:
    """Proactive OBO token-expiry check.

    `auth.py` stashes the JWT `exp` claim on `cl.User.metadata["obo_expires_at"]`
    at handshake time. Reading it per turn is O(1) — no JWT re-decode. Returns
    False (i.e. "not expired") when there's no expiry to check (local dev,
    non-JWT tokens, or `BACKEND_AUTH=sp` where the App-SP's lifecycle is
    managed by the Apps runtime and the OBO `exp` is not the limiting factor),
    letting downstream code take its normal path.
    """
    if os.environ.get("BACKEND_AUTH", "obo").lower() == "sp":
        # SP credentials don't expire from the user's POV — skip the
        # disconnect/reconnect prompt that would mislead the user.
        return False
    session = getattr(cl.context, "session", None)
    user = getattr(session, "user", None) if session else None
    metadata = getattr(user, "metadata", None) or {}
    expires_at = metadata.get("obo_expires_at")
    if expires_at is None:
        return False
    return float(expires_at) < time.time()


def _obo_expiry_seconds() -> tuple[int | None, int | None]:
    """Return (exp_epoch, seconds_to_expiry) from the cached handshake JWT.

    Returns (None, None) when no expiry is stashed (local dev, non-JWT tokens).
    Diagnostic-only — pairs with the `OBO_DIAG handshake` line emitted by
    auth.py so the TTL investigation can correlate handshake-time TTL with
    runway-remaining at each turn.
    """
    session = getattr(cl.context, "session", None)
    user = getattr(session, "user", None) if session else None
    metadata = getattr(user, "metadata", None) or {}
    exp = metadata.get("obo_expires_at")
    if exp is None:
        return None, None
    return int(exp), int(exp) - int(time.time())


@cl.on_message
async def on_message(message: cl.Message):
    backend: Backend | None = cl.user_session.get("backend")
    if backend is None:
        await cl.Message(
            content=("Backend was not initialized — see the error at the top of this chat or reload the page."),
            author="system",
        ).send()
        return

    # Diagnostic — records seconds_to_expiry at the start of every turn.
    # Logged BEFORE the expiry check so we capture the data point even on
    # the turn that triggers the expired-token UX message (that turn is
    # the most informative one — it tells us at exactly what runway-value
    # the proactive check fires vs. when the backend would 403).
    _exp, _seconds_to_expiry = _obo_expiry_seconds()
    logger.info(
        "OBO_DIAG turn now=%s exp=%s seconds_to_expiry=%s",
        int(time.time()), _exp, _seconds_to_expiry,
    )

    # Proactive expiry check — saves ~1-2s of false-hope spinner that would
    # otherwise resolve to a 403 from `/serving-endpoints/responses`. The
    # SDK 403 handler below stays as a safety net for cases where the JWT
    # isn't readable or expiry timing is off.
    if _obo_expired():
        await cl.Message(
            content=(
                "**Your authentication token has expired.** "
                "Please **disconnect and reconnect** (top-right of the chat) to "
                "refresh your session and continue chatting."
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
    except openai.PermissionDeniedError as exc:
        # 403 from /serving-endpoints/responses. Most common cause is an
        # expired OBO token (captured at WebSocket handshake, ~60 min TTL —
        # not refreshed for the lifetime of the session). A simple page
        # reload doesn't help (Chainlit's auth cookie persists); the user
        # needs to disconnect/reconnect via the Chainlit top-right control
        # to force a new handshake and a fresh token. If reconnection
        # doesn't fix it, the user genuinely lacks CAN_QUERY on the
        # endpoint and needs an admin grant.
        await cl.Message(
            content=(
                f"**Permission denied** querying the endpoint: `{exc}`\n\n"
                "Try **disconnecting and reconnecting** (top-right of the chat) — "
                "that refreshes your authentication token. If the issue persists "
                "after reconnecting, contact your workspace admin to confirm you "
                "have access to this endpoint."
            ),
            author="system",
        ).send()
        return
    except Exception as exc:
        await cl.Message(content=f"Backend error: `{type(exc).__name__}: {exc}`").send()
        raise
    finally:
        await renderer.finalize()

    history.append({"role": "assistant", "content": "".join(final_text_parts)})
    cl.user_session.set("history", history)
