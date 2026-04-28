"""Normalize Responses-API stream events into a small set of shapes.

Inspired by `_references/bi-hub-app/src/app/services/mas_normalizer.py` — the
indirection lets the renderer ignore backend-specific quirks (gpt-oss
structured content, attribute-vs-dict shapes, item-typed `task_continue_request`,
etc.). Output shapes consumed by `services.renderer.ChainlitStream`:

    {"type": "message.start", "item_id": str}
    {"type": "text.delta",    "item_id": str, "delta": str}
    {"type": "tool.call",     "call_id": str, "name": str, "args": str}
    {"type": "tool.output",   "call_id": str, "output": str}
    {"type": "thought",       "text": str}

`message.start` + the `item_id` on `text.delta` enable the renderer to keep
each Supervisor message item in its own chronological bubble (see annex §17
of the design doc — `creative_phase_2026-04-27_dbx_apps_streaming_agents.md`).

Anything else (response.created/.completed/etc.) is filtered.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterable, AsyncIterator


def _read(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field whether the object is attribute-style or dict-style."""
    if isinstance(obj, dict):
        val = obj.get(name)
    else:
        val = getattr(obj, name, None)
    return default if val is None else val


def _parse_structured_delta(delta: Any) -> list[dict] | None:
    """Detect gpt-oss-style `[{type: ...}, ...]` payloads inside a text delta.

    Returns the parsed list or None when the delta is plain text.
    """
    if not isinstance(delta, str):
        return None
    s = delta.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return None
    try:
        parsed = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, list) or not parsed:
        return None
    if not all(isinstance(item, dict) and "type" in item for item in parsed):
        return None
    return parsed


def _extract_reasoning_text(item: Any) -> str:
    """Pull human-readable text out of a reasoning item, across known shapes."""
    summary = _read(item, "summary")
    if isinstance(summary, list) and summary:
        first = summary[0]
        if isinstance(first, dict):
            return first.get("text", "") or ""
        return str(first)
    if isinstance(summary, str):
        return summary
    return _read(item, "content", "") or ""


async def normalize(events: AsyncIterable[Any]) -> AsyncIterator[dict]:
    # item_id → item_type, populated from `response.output_item.added` so we
    # can classify deltas whose own event payload doesn't carry the type.
    item_types: dict[str, str] = {}

    async for evt in events:
        evt_type = _read(evt, "type")

        # ---- output item lifecycle ------------------------------------
        # Emit `message.start` immediately on `output_item.added(type=message)`
        # so the renderer can lock the bubble's chat-thread position BEFORE
        # the first delta arrives — sidesteps the §17.3 send-order race.
        if evt_type == "response.output_item.added":
            item = _read(evt, "item")
            if item is not None:
                item_id = _read(item, "id", "") or ""
                item_type = _read(item, "type", "") or ""
                if item_id:
                    item_types[item_id] = item_type
                if item_type == "message" and item_id:
                    yield {"type": "message.start", "item_id": item_id}
            continue

        # ---- text deltas -----------------------------------------------
        if evt_type == "response.output_text.delta":
            delta = _read(evt, "delta", "")
            if not delta:
                continue
            # `item_id` lives on the delta event itself per the Responses API
            # contract (not nested under `item`).
            item_id = _read(evt, "item_id", "") or ""
            structured = _parse_structured_delta(delta)
            if structured is None:
                yield {"type": "text.delta", "item_id": item_id, "delta": delta}
                continue
            # gpt-oss structured content — split into reasoning + text
            for sub in structured:
                sub_type = sub.get("type")
                if sub_type == "reasoning":
                    text = _extract_reasoning_text(sub)
                    if text:
                        yield {"type": "thought", "text": text}
                elif sub_type == "text":
                    text = sub.get("text", "")
                    if text:
                        yield {"type": "text.delta", "item_id": item_id, "delta": text}
                # other sub-types (image, etc.) are dropped here
            continue

        # ---- finalized output items ------------------------------------
        if evt_type == "response.output_item.done":
            item = _read(evt, "item")
            if item is None:
                continue
            item_type = _read(item, "type")

            if item_type == "function_call":
                call_id = _read(item, "call_id", "") or _read(item, "id", "")
                yield {
                    "type": "tool.call",
                    "call_id": call_id,
                    "name": _read(item, "name", "tool"),
                    "args": _read(item, "arguments", "") or "",
                }
                continue

            if item_type == "function_call_output":
                call_id = _read(item, "call_id", "") or _read(item, "id", "")
                yield {
                    "type": "tool.output",
                    "call_id": call_id,
                    "output": _read(item, "output", "") or "",
                }
                continue

            if item_type == "reasoning":
                text = _extract_reasoning_text(item)
                if text:
                    yield {"type": "thought", "text": text}
                continue

            # `message`, `task_continue_request`, other item types: ignored
            continue

        # response.created / .in_progress / .completed / task_continue_request: ignored
