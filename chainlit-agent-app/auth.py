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
import logging
import time
from typing import Any, Dict, Optional

import chainlit as cl


# Diagnostic logger for the OBO TTL investigation — see
# memory_bank/creative_phases/creative_phase_2026-05-07_chainlit_obo_expiry.md.
# Emits one `OBO_DIAG handshake` line per WebSocket auth (here in auth.py)
# and one `OBO_DIAG turn` line per chat message (in app.py). Lines deliberately
# carry only timestamps and claim values — never token bytes.
logger = logging.getLogger("obo_diag")


def _decode_jwt_claims(token: str) -> Dict[str, Optional[int]]:
    """Decode a JWT's payload (unsigned, unverified) and return its `iat` and
    `exp` claims as epoch seconds. Returns {iat: None, exp: None} if the token
    isn't a JWT or lacks claims. We don't verify the signature — Databricks
    does that server-side; we just need timestamps for diagnostics + proactive
    UX. `iat` enables computing the design TTL (`exp - iat`); without it we
    can only see runway-remaining, not what the runway-total was meant to be.
    """
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return {"iat": None, "exp": None}
        # JWT base64url payload may lack padding; pad to length % 4 == 0.
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        payload: dict = json.loads(base64.urlsafe_b64decode(payload_b64))
        iat = payload.get("iat")
        exp = payload.get("exp")
        return {
            "iat": int(iat) if iat is not None else None,
            "exp": int(exp) if exp is not None else None,
        }
    except Exception:
        return {"iat": None, "exp": None}


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
    # Stash claims once at handshake so on_message can do an O(1) check
    # without re-decoding the JWT each turn. Decoder returns
    # {iat:None, exp:None} for non-JWT tokens (e.g. PAT-style strings); in
    # that case expiry-check is skipped and we fall through to the SDK's
    # 403 handler. `obo_issued_at` is required for the TTL measurement
    # (`exp - iat`); see creative_phase_2026-05-07_chainlit_obo_expiry.md.
    claims = _decode_jwt_claims(token)
    iat, exp = claims["iat"], claims["exp"]
    if exp is not None:
        metadata["obo_expires_at"] = exp
    if iat is not None:
        metadata["obo_issued_at"] = iat
    ttl = (exp - iat) if (iat is not None and exp is not None) else None
    logger.info(
        "OBO_DIAG handshake user=%s iat=%s exp=%s ttl_seconds=%s now=%s",
        email or "unknown", iat, exp, ttl, int(time.time()),
    )

    return cl.User(
        identifier=email or "unknown",
        display_name=(email or "user").split("@")[0],
        email=email,
        provider="obo",
        metadata=metadata,
    )
