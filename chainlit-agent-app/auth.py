"""Chainlit header-auth — captures Databricks Apps OBO token and binds it to
the chat session for per-user identity at the serving endpoint.

How it works
------------
Databricks Apps injects the browsing user's bearer token in
`x-forwarded-access-token` when `databricks.yml` declares
`user_api_scopes: ["serving.serving-endpoints"]`. Chainlit invokes
`auth_from_header` on each WebSocket handshake; we extract the token and
stash it on `cl.User.metadata["obo_token"]`. `app.py:_obo_token_from_session()`
reads it back per chat turn and threads it into `EndpointBackend.from_env(
obo_token=...)`, which constructs `WorkspaceClient(token=obo_token,
auth_type="pat")`.

The `auth_type="pat"` is required because the Apps runtime auto-injects
OAuth client credentials (`DATABRICKS_CLIENT_ID` / `DATABRICKS_CLIENT_SECRET`)
for the App's own service principal, and on top of that we pass an
explicit user-bearer — the SDK sees two auth methods configured and
refuses to pick. Setting `auth_type="pat"` says "ignore the OAuth env
vars, treat my explicit token as a PAT bearer". `host=` is required for
the same reason: opting out of the env-var chain that would otherwise
pick it up.

Why OBO over App-SP
-------------------
Two reasons: **permission ergonomics** and **observability**.

*Ergonomics.* Running as the App SP requires explicit `CAN_QUERY` grants
for that SP on the target serving endpoint AND, for AgentBricks
Supervisor, likely on every sub-agent it routes to (KAs, Genies, UC
functions). Each new workspace needs a parallel grant pass against the
App's SP. With OBO, the calling user's permissions are checked at
request time — any user who can use the Supervisor via the workspace UI
can use it via the App without additional admin work.

*Observability.* The App-SP / OAuth-M2M path against an AgentBricks
Supervisor has an observed footgun: when the App SP appears to lack
adequate grants, the symptom is **HTTP 200 OK with zero SSE events**
(the iterator drains empty, no exception fires, the UI shows nothing).
Root cause is not pinned at the wire level — possible explanations
include the M2M auth path's error handling differing from user-bearer,
a Supervisor-internal pre-validation that swallows errors when the caller is a service principal, or a missing sub-agent grant that the
orchestrator handles silently. The OBO path under the same logical
condition (user lacks grants) returns a clean **4xx Not Authorized**
that propagates as a normal exception. OBO isn't just per-user
identity — it's the path with usable error behavior.

Verified against /serving-endpoints/responses
---------------------------------------------
Older reference apps documented that `POST /serving-endpoints/responses`
(the OpenAI-compatible Responses API path that `AsyncDatabricksOpenAI`
uses) returned `403 Forbidden: Invalid Token` for OBO-derived bearers,
requiring a fallback to `POST /serving-endpoints/<name>/invocations`
with raw httpx + SSE parsing. **That claim is stale on current
Databricks runtime.** OBO bearers issued via `x-forwarded-access-token`
reach `/responses` successfully and stream Responses-API events through
`AsyncDatabricksOpenAI` end-to-end. Single transport, single mental model.

Local dev fallback
------------------
When no `x-forwarded-access-token` header is present (i.e. `chainlit run`
locally with no Apps proxy in front), we return a placeholder `cl.User`
rather than `None`. Returning `None` would 401 the WebSocket handshake
and make local dev impossible. Downstream, `EndpointBackend.from_env(
obo_token=None)` then resolves auth via the standard WorkspaceClient
chain (DEFAULT profile / `DATABRICKS_CONFIG_PROFILE` / explicit env
vars), so local development auths via PAT or a named profile.

`CHAINLIT_AUTH_SECRET` must be set (any non-empty string in `.env` for
local dev; rotate to a real secret in `app.yaml` for deployed Apps —
generate via `python -c "import secrets; print(secrets.token_urlsafe(32))"`).
The decorator silently fails to register without one.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, Optional

import chainlit as cl


def _decode_jwt_exp(token: str) -> Optional[int]:
    """Decode a JWT's payload (unsigned, unverified) and return its `exp`
    claim as epoch seconds. Returns None if the token isn't a JWT or has no
    `exp`. We don't verify the signature — Databricks does that server-side;
    we just need the expiry timestamp for proactive UX. Same pattern bi-hub
    apps use for OBO TTL inspection.
    """
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return None
        # JWT base64url payload may lack padding; pad to length % 4 == 0.
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        payload: dict = json.loads(base64.urlsafe_b64decode(payload_b64))
        exp = payload.get("exp")
        return int(exp) if exp is not None else None
    except Exception:
        return None


@cl.header_auth_callback
def auth_from_header(headers: Dict[str, str]) -> Optional[cl.User]:
    """Extract OBO token from `x-forwarded-access-token` and bind to a `cl.User`."""
    token = headers.get("x-forwarded-access-token")
    email = headers.get("x-forwarded-email") or headers.get("x-forwarded-user")

    if not token:
        # Local dev: no Apps proxy → no OBO header. Return a placeholder so
        # Chainlit's WebSocket handshake doesn't 401. With no obo_token in
        # metadata, EndpointBackend.from_env(obo_token=None) falls back to
        # the standard WorkspaceClient auth chain.
        return cl.User(identifier="local-dev", metadata={"auth_type": "local"})

    metadata: Dict[str, Any] = {"auth_type": "obo", "obo_token": token}
    # Stash expiry once at handshake so on_message can do an O(1) check
    # without re-decoding the JWT each turn. Decoder returns None for
    # non-JWT tokens (e.g. PAT-style strings); in that case, expiry-check
    # is skipped and we fall through to the SDK's 403 handler.
    exp = _decode_jwt_exp(token)
    if exp is not None:
        metadata["obo_expires_at"] = exp

    return cl.User(
        identifier=email or "unknown",
        display_name=(email or "user").split("@")[0],
        email=email,
        provider="obo",
        metadata=metadata,
    )
