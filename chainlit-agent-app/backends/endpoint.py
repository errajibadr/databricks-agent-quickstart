"""Endpoint backend — streams from a deployed Databricks Serving Endpoint.

Mirrors `LocalAgentBackend` shape but talks to the workspace over the wire via
`AsyncDatabricksOpenAI`. The same Chainlit handler in `app.py` consumes this
backend; only the construction site differs.

Auth model
----------
`AsyncDatabricksOpenAI` (databricks_openai/utils/clients.py) wires
`BearerAuth(workspace_client.config.authenticate)` as an httpx auth plugin —
the bearer is fetched **per request** off the WorkspaceClient. That single
property is what lets one client class cover three deployment shapes:

    Local laptop      WorkspaceClient()                # DEFAULT profile / .env
    Deployed Apps     WorkspaceClient(token=obo_token) # x-forwarded-access-token (per-user identity)
    Deployed Apps     WorkspaceClient()                # App-SP from DATABRICKS_CLIENT_ID/SECRET

`BACKEND_AUTH` env var picks between OBO (default, per-user identity at the
endpoint) and SP (App service principal credentials). Use SP when:

  * Sessions need to outlive OBO TTL (~60 min on Azure Databricks) — App-SP
    credentials are managed by the Apps runtime and don't expire from the
    user's POV, so no Chainlit reconnect (and conversation-state loss) is
    needed mid-session.
  * Per-user identity at the endpoint isn't required and simpler grants are
    preferred (a single `CAN_QUERY` to the App's SP vs. per-user grants).

Trade-offs of SP mode:

  * Audit at the serving endpoint shows the App SP, not the calling user.
  * Sub-agents (KAs, Genie, UC functions) for AgentBricks Supervisor each
    need explicit grants to the App SP — separate admin pass per workspace.
  * Watch out for the 200-OK-zero-events footgun documented in `auth.py`:
    when the App SP appears to lack adequate grants on a Supervisor's sub-
    agents, the symptom can be HTTP 200 + empty SSE stream rather than a
    clean 4xx. Validate grants explicitly before recommending SP for
    production.

`from_env(obo_token=...)` is the single construction surface. Pass the OBO
token only when `BACKEND_AUTH=obo` and a forwarded token is available;
otherwise leave it None and the SDK's auth chain picks creds (DEFAULT
profile locally, App SP in deployed Apps).
"""

from __future__ import annotations

import os
import time
from typing import Any, AsyncIterator

from databricks.sdk import WorkspaceClient
from databricks_openai import AsyncDatabricksOpenAI

_LOG_EVENTS = os.environ.get("DBX_AGENT_LOG_EVENTS", "1") not in {"0", "false", ""}


class EndpointBackend:
    """Backend Protocol shape: yields native `ResponseStreamEvent` objects.

    The client is built once per Chainlit chat session (constructed in
    `on_chat_start`). For Lane L2/L4 the WorkspaceClient resolves auth via
    the standard Databricks chain (DEFAULT profile, `DATABRICKS_CONFIG_PROFILE`,
    explicit host/token env vars, etc.). For Lane L3 the App handler should
    construct the backend with `from_env(obo_token=request_header)` per session.
    """

    def __init__(self, endpoint_name: str, workspace_client: WorkspaceClient):
        self.endpoint = endpoint_name
        self.client = AsyncDatabricksOpenAI(workspace_client=workspace_client)

    @classmethod
    def from_env(cls, obo_token: str | None = None) -> "EndpointBackend":
        endpoint_name = os.environ.get("ENDPOINT_NAME")
        if not endpoint_name:
            raise RuntimeError("ENDPOINT_NAME must be set for BACKEND=endpoint. Example: ENDPOINT_NAME=doc-agent-quickstart")

        # `obo` (default) → use the forwarded user token if provided, else fall
        # back to the SDK auth chain (DEFAULT profile locally, App SP in Apps).
        # `sp` → ignore any forwarded token; always use the SDK auth chain so
        # in deployed Apps we land on App-SP credentials. See module docstring
        # for the trade-offs.
        backend_auth = os.environ.get("BACKEND_AUTH", "obo").lower()
        if backend_auth not in {"obo", "sp"}:
            raise RuntimeError(f"BACKEND_AUTH must be 'obo' or 'sp', got {backend_auth!r}.")

        if backend_auth == "obo" and obo_token:
            ws = WorkspaceClient(
                host=os.environ.get("DATABRICKS_HOST"),
                token=obo_token,
                auth_type="pat",
            )
        else:
            ws = WorkspaceClient()

        return cls(endpoint_name, ws)

    async def stream(self, messages: list[dict]) -> AsyncIterator[Any]:
        """Yield native Responses-API events; `event_normalizer.py` shapes them."""
        start = time.monotonic()
        if _LOG_EVENTS:
            print(f"[backend] +0.000s  STREAM_START  endpoint={self.endpoint}", flush=True)

        # `responses.create(stream=True)` returns an awaitable that resolves to
        # an async iterator of ResponseStreamEvent. The two-step await/iterate
        # is the OpenAI SDK's standard streaming shape.
        stream = await self.client.responses.create(
            model=self.endpoint,
            input=messages,
            stream=True,
        )

        try:
            async for event in stream:
                if _LOG_EVENTS:
                    evt_type = getattr(event, "type", "?")
                    item = getattr(event, "item", None)
                    item_type = getattr(item, "type", "") if item is not None else ""
                    suffix = f"  [item={item_type}]" if item_type else ""
                    print(
                        f"[backend] +{time.monotonic() - start:6.3f}s  {evt_type}{suffix}",
                        flush=True,
                    )
                yield event
        finally:
            if _LOG_EVENTS:
                print(
                    f"[backend] +{time.monotonic() - start:6.3f}s  STREAM_END",
                    flush=True,
                )
