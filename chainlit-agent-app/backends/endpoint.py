"""Endpoint backend — streams from a deployed Databricks Serving Endpoint.

Mirrors `LocalAgentBackend` shape but talks to the workspace over the wire via
`AsyncDatabricksOpenAI`. The same Chainlit handler in `app.py` consumes this
backend; only the construction site differs.

Auth model
----------
`AsyncDatabricksOpenAI` (databricks_openai/utils/clients.py) wires
`BearerAuth(workspace_client.config.authenticate)` as an httpx auth plugin —
the bearer is fetched **per request** off the WorkspaceClient. That single
property is what lets one client class cover three deployment lanes:

    Lane L2 (local PAT)        WorkspaceClient()                # DEFAULT profile / .env
    Lane L3 (Apps OBO)         WorkspaceClient(token=obo_token) # x-forwarded-access-token
    Lane L4 (DACHSER Citrix)   WorkspaceClient()                # workspace-resolved auth

`from_env(obo_token=...)` is the single construction surface; pass the OBO
token only on the Apps deployment path. Local dev passes nothing and lets
the SDK's auth chain pick a profile.
"""

from __future__ import annotations

import os
import time
from typing import Any, AsyncIterator

from databricks.sdk import WorkspaceClient
from databricks_openai import AsyncDatabricksOpenAI

# Same diagnostic gate as `local_agent.py`: leave on during Step B bring-up
# so the first-call cold-start latency profile is visible. Flip to "0" once
# the pre-warm pattern lands.
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
            raise RuntimeError(
                "ENDPOINT_NAME must be set for BACKEND=endpoint. "
                "Example: ENDPOINT_NAME=doc-agent-quickstart"
            )

        if obo_token:
            # Apps runtime auto-injects DATABRICKS_CLIENT_ID / CLIENT_SECRET
            # for the App's own SP. With our explicit OBO token, that's two
            # auth methods configured at once → SDK refuses to pick (raises
            # "more than one authorization method configured"). `auth_type="pat"`
            # tells the SDK "ignore the OAuth env vars, treat my explicit
            # token as a PAT". `host=` is required because we're now opting
            # out of the env-var chain that would have picked it up.
            ws = WorkspaceClient(
                host=os.environ.get("DATABRICKS_HOST"),
                token=obo_token,
                auth_type="pat",
            )
        else:
            # Local path: WorkspaceClient walks the standard Databricks unified
            # auth chain (DEFAULT profile / .env / SP).
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
