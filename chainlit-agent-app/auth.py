"""Chainlit header-auth — reads OBO token from Databricks Apps request headers.

Databricks Apps forwards the user's bearer token in `x-forwarded-access-token`
when the App declares `user_api_scopes: ["serving.serving-endpoints"]` in its
DABs config. Chainlit invokes `@cl.header_auth_callback` once at WebSocket
upgrade time — that's the only place those headers are reachable. We stash the
token on `cl.User.metadata["obo_token"]` so `app.py:_build_backend()` can pull
it back when constructing `EndpointBackend`.

Activation: only fires when `CHAINLIT_AUTH_SECRET` is set in the environment
(Chainlit treats absence as "anonymous mode" and skips the callback). Locally
we leave it unset — `cl.context.session.user` is None and the backend falls
back to the standard WorkspaceClient auth chain. In Apps it's set in app.yaml.

Reference: `_references/bi-hub-app/src/app/auth/header.py`. Stripped down here:
no JWT expiry check (Apps refreshes per request), no email gating, single
purpose — token capture.
"""

from __future__ import annotations

from typing import Dict, Optional

import chainlit as cl


@cl.header_auth_callback
def auth_from_header(headers: Dict[str, str]) -> Optional[cl.User]:
    """Extract OBO token from `x-forwarded-access-token` and bind to a `cl.User`.

    Returning `None` rejects the request — Apps will surface a 401 to the
    browser. Always-return-User is intentional: in a Databricks App context
    the absence of the header is itself a misconfiguration we want loud.
    """
    token = headers.get("x-forwarded-access-token")
    email = headers.get("x-forwarded-email") or headers.get("x-forwarded-user")

    if not token:
        # No header → not an Apps deployment, or a misrouted request. Reject
        # rather than silently fall through to anonymous (which would mask the
        # config error until the first endpoint call 401s with a worse trace).
        return None

    return cl.User(
        identifier=email or "unknown",
        display_name=(email or "user").split("@")[0],
        email=email,
        provider="obo",
        metadata={"auth_type": "obo", "obo_token": token},
    )
