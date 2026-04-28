"""Chainlit header-auth — placeholder for future per-user OBO capture.

Why this is currently un-registered
-----------------------------------
Databricks Apps forwards the browsing user's bearer token in
`x-forwarded-access-token` when the App declares `user_api_scopes:
["serving.serving-endpoints"]`. We *could* register `@cl.header_auth_callback`
to capture it and route the token through `EndpointBackend(WorkspaceClient(
token=obo_token, auth_type="pat"))`. That gives true per-user identity at
the serving endpoint — the right design for multi-user / audit-required
deployments like DACHSER.

**It does not currently work against AsyncDatabricksOpenAI.** The OBO token
is correctly captured and forwarded, but `POST /serving-endpoints/responses`
(the OpenAI-compatible Responses API path that `AsyncDatabricksOpenAI` uses)
returns `403 Forbidden: Invalid Token` for OBO-derived bearers. The same
token works against the per-endpoint `POST /serving-endpoints/<name>/invocations`
path — see `_references/bi-hub-app/src/app/services/mas_client.py:118`
(`_stream_rest_sse` — "needed for OBO"). bi-hub-app uses raw httpx + SSE
parsing for the OBO path and `AsyncOpenAI` only for PAT.

**With no callback registered**, Chainlit treats every session as anonymous,
`cl.context.session.user` is None, `_obo_token_from_session()` returns None,
and `EndpointBackend.from_env(obo_token=None)` falls back to the standard
WorkspaceClient auth chain. In Apps that means the App's own service
principal (auto-injected via `DATABRICKS_CLIENT_ID/SECRET`) is used —
which works because `databricks.yml` grants the SP `CAN_QUERY` on the
endpoint. It is NOT per-user OBO; for personal sandbox testing that's
fine, for DACHSER it isn't.

To enable real OBO when ready
-----------------------------
1. Add `@cl.header_auth_callback` to `auth_from_header` below (and set
   `CHAINLIT_AUTH_SECRET` in both `.env` and `app.yaml`).
2. Add an OBO branch in `backends/endpoint.py:stream` that posts to
   `/serving-endpoints/<name>/invocations` with raw SSE, mirroring
   bi-hub-app's `MASChatClient._stream_rest_sse`. Keep `AsyncDatabricksOpenAI`
   for the PAT / SP path.
3. Update the `local-dev` placeholder return below — Chainlit raises 401
   on `None`, so locally we'd need to return a placeholder User.

Reference: `_references/bi-hub-app/src/app/auth/header.py` (full pattern).
"""

from __future__ import annotations

from typing import Dict, Optional

import chainlit as cl


# NOT decorated with @cl.header_auth_callback — see module docstring.
# Once the OBO transport in `backends/endpoint.py` switches to /invocations
# SSE, restore the decorator and the local-dev placeholder return path.
def auth_from_header(headers: Dict[str, str]) -> Optional[cl.User]:
    """Extract OBO token from `x-forwarded-access-token` and bind to a `cl.User`."""
    token = headers.get("x-forwarded-access-token")
    email = headers.get("x-forwarded-email") or headers.get("x-forwarded-user")

    if not token:
        return None

    return cl.User(
        identifier=email or "unknown",
        display_name=(email or "user").split("@")[0],
        email=email,
        provider="obo",
        metadata={"auth_type": "obo", "obo_token": token},
    )
