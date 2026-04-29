"""Endpoint backend — streams from a deployed Databricks Serving Endpoint.

Mirrors `LocalAgentBackend` shape but talks to the workspace over the wire via
`AsyncDatabricksOpenAI`. The same Chainlit handler in `app.py` consumes this
backend; only the construction site differs.

Auth model
----------
`AsyncDatabricksOpenAI` (databricks_openai/utils/clients.py) wires
`BearerAuth(workspace_client.config.authenticate)` as an httpx auth plugin —
the bearer is fetched **per request** off the WorkspaceClient. That single
property is what lets one client class cover both deployment shapes:

    Local laptop      WorkspaceClient()                # DEFAULT profile / .env
    Deployed Apps     WorkspaceClient(token=obo_token) # x-forwarded-access-token

`from_env(obo_token=...)` is the single construction surface; pass the OBO
token only on the Apps deployment path. Local dev passes nothing and lets
the SDK's auth chain pick a profile.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, AsyncIterator

import httpx
from databricks.sdk import WorkspaceClient
from databricks_openai import AsyncDatabricksOpenAI

# Same diagnostic gate as `local_agent.py`: leave on during Step B bring-up
# so the first-call cold-start latency profile is visible. Flip to "0" once
# the pre-warm pattern lands.
_LOG_EVENTS = os.environ.get("DBX_AGENT_LOG_EVENTS", "1") not in {"0", "false", ""}

# Diagnostic probe: when DBX_PROBE_RAW=1, stream() bypasses the SDK and does
# a raw httpx POST to the same /serving-endpoints/responses URL the SDK would
# hit, logging full status / headers / body so we can see what the endpoint
# actually returns when the SDK iterator drains empty. Yields zero events in
# this mode — diagnostic-only. Toggle off to restore normal streaming.
_PROBE_RAW = os.environ.get("DBX_PROBE_RAW", "1") not in {"0", "false", ""}


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
        # Held for the probe path (DBX_PROBE_RAW=1) which needs config.host
        # and config.authenticate() to build the raw httpx request.
        self.workspace_client = workspace_client
        self.client = AsyncDatabricksOpenAI(workspace_client=workspace_client)

    @classmethod
    def from_env(cls, obo_token: str | None = None) -> "EndpointBackend":
        endpoint_name = os.environ.get("ENDPOINT_NAME")
        if not endpoint_name:
            raise RuntimeError("ENDPOINT_NAME must be set for BACKEND=endpoint. Example: ENDPOINT_NAME=doc-agent-quickstart")

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

    async def _probe_raw_response(self, messages: list[dict]) -> None:
        """Diagnostic: do a raw httpx POST to /serving-endpoints/responses and
        log everything that comes back. No event yielded — populates App logs
        so we can see what the endpoint actually returns (status, headers,
        body) when the SDK iterator drains empty. Triggered by DBX_PROBE_RAW=1.

        Critical: `config.authenticate()` returns Dict[str, str] of HTTP
        headers — not a token string. On Azure SP it can carry extras like
        `X-Databricks-Azure-SP-Management-Token`. Merge the whole dict.
        """
        start = time.monotonic()
        config = self.workspace_client.config

        host = (config.host or "").rstrip("/")
        if not host:
            print("[probe] ERROR  workspace_client.config.host is empty; cannot probe", flush=True)
            return

        url = f"{host}/serving-endpoints/responses"
        body = {"model": self.endpoint, "input": messages, "stream": True}

        try:
            auth_headers = config.authenticate()
        except Exception as exc:
            print(f"[probe] ERROR  resolving auth headers: {type(exc).__name__}: {exc}", flush=True)
            return

        request_headers = dict(auth_headers)
        request_headers["Content-Type"] = "application/json"
        request_headers["Accept"] = "text/event-stream"

        body_preview = json.dumps(body)
        if len(body_preview) > 2000:
            body_preview = body_preview[:2000] + f"...(truncated, total={len(body_preview)} chars)"

        print(f"[probe] +0.000s  REQUEST  POST {url}", flush=True)
        print(f"[probe] +0.000s  body  {body_preview}", flush=True)
        print(f"[probe] +0.000s  auth_header_names  {sorted(auth_headers.keys())}", flush=True)

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", url, headers=request_headers, json=body) as resp:
                    t_headers = time.monotonic() - start
                    print(f"[probe] +{t_headers:6.3f}s  STATUS  {resp.status_code} {resp.reason_phrase}", flush=True)
                    for k, v in resp.headers.items():
                        print(f"[probe] +{t_headers:6.3f}s  header  {k}: {v}", flush=True)

                    content_type = resp.headers.get("content-type", "").lower()

                    if "event-stream" in content_type:
                        line_count = 0
                        data_count = 0
                        total_bytes = 0
                        async for line in resp.aiter_lines():
                            line_count += 1
                            total_bytes += len(line.encode("utf-8")) if line else 0
                            elapsed = time.monotonic() - start
                            if line.startswith("data:"):
                                data_count += 1
                                payload = line[5:].strip()
                                snippet = payload if len(payload) <= 500 else payload[:500] + f"...(+{len(payload) - 500} chars)"
                                print(f"[probe] +{elapsed:6.3f}s  data#{data_count}  {snippet}", flush=True)
                            elif line.startswith(":"):
                                print(f"[probe] +{elapsed:6.3f}s  comment  {line[:200]}", flush=True)
                            elif line:
                                print(f"[probe] +{elapsed:6.3f}s  raw  {line[:200]}", flush=True)
                            # blank lines silently skipped (SSE frame separators)
                        print(
                            f"[probe] +{time.monotonic() - start:6.3f}s  SSE_SUMMARY  lines={line_count}  data_events={data_count}  bytes={total_bytes}",
                            flush=True,
                        )

                    elif "json" in content_type:
                        body_bytes = await resp.aread()
                        elapsed = time.monotonic() - start
                        print(f"[probe] +{elapsed:6.3f}s  JSON_BODY  bytes={len(body_bytes)}", flush=True)
                        try:
                            text = body_bytes.decode("utf-8")
                        except UnicodeDecodeError:
                            text = repr(body_bytes[:2000])
                        # Chunk into ~1000-char lines so platform log limits don't truncate.
                        for i in range(0, len(text), 1000):
                            print(f"[probe] +{elapsed:6.3f}s  body[{i}:{i + 1000}]  {text[i : i + 1000]}", flush=True)

                    else:
                        body_bytes = await resp.aread()
                        elapsed = time.monotonic() - start
                        print(
                            f"[probe] +{elapsed:6.3f}s  OTHER  content-type={content_type!r}  bytes={len(body_bytes)}",
                            flush=True,
                        )
                        if body_bytes:
                            snippet = body_bytes[:256]
                            print(f"[probe] +{elapsed:6.3f}s  hex_first_256  {snippet.hex(' ')}", flush=True)
                            print(
                                f"[probe] +{elapsed:6.3f}s  utf8_first_256  {snippet.decode('utf-8', errors='replace')}",
                                flush=True,
                            )
                        else:
                            print(f"[probe] +{elapsed:6.3f}s  EMPTY_BODY", flush=True)

        except Exception as exc:
            elapsed = time.monotonic() - start
            print(f"[probe] +{elapsed:6.3f}s  TRANSPORT_ERROR  {type(exc).__name__}: {exc}", flush=True)

        print(f"[probe] +{time.monotonic() - start:6.3f}s  PROBE_END", flush=True)

    async def stream(self, messages: list[dict]) -> AsyncIterator[Any]:
        """Yield native Responses-API events; `event_normalizer.py` shapes them."""
        if _PROBE_RAW:
            # Diagnostic-only path: log raw HTTP response, yield nothing.
            # Toggle DBX_PROBE_RAW=0 to restore normal SDK-streaming behavior.
            print(f"[backend] PROBE_MODE  endpoint={self.endpoint}  (yielding zero events)", flush=True)
            await self._probe_raw_response(messages)
            return

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
        print(stream)

        try:
            async for event in stream:
                print(event)
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
