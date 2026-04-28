"""ChainlitStream — chronological per-item rendering for Responses-API events.

Each output item from the agent gets its own UI element in chat-thread order.
Replaces the bi-hub-app-style status-aggregator + text-bubble pattern from §16
(which mis-handled Supervisor multi-message turns — see §17 of the design
annex `creative_phase_2026-04-27_dbx_apps_streaming_agents.md`).

Flow:

    message.start (item_id)        → cl.Message(author=assistant), empty, sent
                                     immediately to lock its chat-thread position
    text.delta   (item_id, delta) → stream_token into the matching bubble;
                                     lazy-create when the agent didn't emit
                                     `output_item.added` (e.g. older shapes)
    tool.call    (call_id, ...)   → cl.Message styled with <details open>
                                     showing the tool name + args
    tool.output  (call_id, ...)   → re-render that same message with output
                                     and a closed <details>
    thought      (text)            → cl.Message(author=thinking), collapsed

Why `cl.Message` styled with <details> instead of `cl.Step` for tools:
Chainlit issue #2365 (open as of 2.11.x) renders Steps below the parent
assistant message rather than in chronological send-order. Using a plain
`cl.Message` per tool entry preserves event-order positioning.
"""

from __future__ import annotations

import html
from typing import Any

import chainlit as cl


_RESULT_PREVIEW_LIMIT = 800
_THOUGHT_INLINE_LIMIT = 140
_EMPTY_ARG_VALUES = {"{}", "[]", ""}


def _is_meaningful(value: str) -> bool:
    """True when a string has substantive content beyond empty-JSON placeholders."""
    return bool(value) and value.strip() not in _EMPTY_ARG_VALUES


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…\n_(truncated; {len(text) - limit} chars hidden)_"


def _escape_inline(text: str) -> str:
    """Escape backticks for inline-code rendering."""
    return text.replace("`", "\\`")


class ChainlitStream:
    def __init__(self) -> None:
        # item_id → bubble for streaming text. Lazy-created on first delta when
        # `message.start` wasn't emitted (Lane L1 / older agent event shapes).
        self._messages: dict[str, cl.Message] = {}
        # call_id → {message: cl.Message, name, args, output, running}
        self._tool_entries: dict[str, dict[str, Any]] = {}

    # ---- public event hooks (consumed by app.py) -----------------------

    async def on_message_start(self, item_id: str) -> None:
        """Pre-create an empty assistant bubble in send-order (chronological lock)."""
        if not item_id or item_id in self._messages:
            return
        msg = cl.Message(content="", author="assistant")
        await msg.send()
        self._messages[item_id] = msg

    async def on_text_delta(self, item_id: str, token: str) -> None:
        if not token:
            return
        msg = self._messages.get(item_id)
        if msg is None:
            # Agent skipped `output_item.added` (Lane L1's `predict_stream`
            # may not emit it). Create lazily so we don't drop tokens.
            msg = cl.Message(content="", author="assistant")
            await msg.send()
            # Empty item_id is fine as a key — only one such bubble per turn.
            self._messages[item_id] = msg
        await msg.stream_token(token)

    async def on_tool_call(self, call_id: str, name: str, args: str) -> None:
        entry: dict[str, Any] = {
            "name": name,
            "args": args or "",
            "output": None,
            "running": True,
        }
        msg = cl.Message(content=self._render_tool(entry), author="tool")
        await msg.send()
        entry["message"] = msg
        # Empty call_id collisions are rare but possible — keep last one.
        self._tool_entries[call_id] = entry

    async def on_tool_output(self, call_id: str, output: str) -> None:
        entry = self._tool_entries.get(call_id)
        if entry is None:
            # Orphan output (no matching call_id seen) — render standalone in
            # chronological position rather than drop it on the floor.
            entry = {
                "name": "tool_result",
                "args": "",
                "output": output or "",
                "running": False,
            }
            msg = cl.Message(content=self._render_tool(entry), author="tool")
            await msg.send()
            entry["message"] = msg
            self._tool_entries[call_id] = entry
            return
        entry["output"] = output or ""
        entry["running"] = False
        msg: cl.Message = entry["message"]
        msg.content = self._render_tool(entry)
        await msg.update()

    async def on_thought(self, text: str) -> None:
        if not text:
            return
        rendered = self._render_thought(text)
        if not rendered:
            return
        msg = cl.Message(content=rendered, author="thinking")
        await msg.send()

    async def finalize(self) -> None:
        """Defensive terminator for any tool entry that never received its output."""
        for entry in self._tool_entries.values():
            if entry["running"]:
                entry["running"] = False
                msg: cl.Message = entry["message"]
                msg.content = self._render_tool(entry)
                await msg.update()

    # ---- markdown renderers --------------------------------------------

    def _render_thought(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        if len(text) <= _THOUGHT_INLINE_LIMIT and "\n" not in text:
            return f"💭 _{text}_"
        first_line = text.split("\n", 1)[0]
        summary_preview = first_line[:_THOUGHT_INLINE_LIMIT]
        if len(first_line) > _THOUGHT_INLINE_LIMIT:
            summary_preview += "…"
        return (
            f"<details><summary>💭 <i>{html.escape(summary_preview)}</i></summary>"
            f"\n\n{text}\n\n</details>"
        )

    def _render_tool(self, entry: dict[str, Any]) -> str:
        name = entry["name"]
        args = (entry.get("args") or "").strip()
        output = entry.get("output")
        running = entry["running"]

        head = (
            f"🛠️ <b>{html.escape(name)}</b> running…"
            if running
            else f"✅ <b>{html.escape(name)}</b> completed"
        )

        sections: list[str] = []
        if _is_meaningful(args):
            sections.append(f"**Args**: `{_escape_inline(args)}`")
        if output is not None and output != "":
            preview = _truncate(output, _RESULT_PREVIEW_LIMIT)
            # Plain code-fence (no language tag) — earlier `text` tag both
            # suppressed markdown rendering AND surfaced "text" as a header
            # badge in Chainlit's code-block component.
            sections.append(f"```\n{preview}\n```")

        if not sections:
            return f"- {head}"

        body = "\n\n".join(sections)
        # Open while running so args are visible without a click; collapse
        # automatically when output arrives.
        open_attr = " open" if running else ""
        return f"<details{open_attr}><summary>{head}</summary>\n\n{body}\n\n</details>"
