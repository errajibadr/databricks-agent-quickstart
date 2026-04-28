"""ChainlitStream — bi-hub-app-style status aggregator + text bubble.

Two persistent Messages per turn:
  - status_msg : a single Message that aggregates 💭 thoughts and 🛠️/✅ tool
                 lines into an ordered activity feed (live-updated in place).
                 Tool entries become <details> expanders showing args + result.
  - text_msg   : the response bubble. Created lazily *after* the status
                 message so it sits below it in the chat thread.

Pattern mirrors `_references/bi-hub-app/src/app/services/renderer.py` but
adds reasoning/thought support for reasoning-capable models (gpt-oss).
"""

from __future__ import annotations

import html
from typing import Any, Optional

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
        self.status_msg: Optional[cl.Message] = None
        self.text_msg: Optional[cl.Message] = None

        # Ordered activity feed: each entry is one of:
        #   {"kind": "thought", "text": str}
        #   {"kind": "tool", "call_id": str, "name": str, "args": str,
        #    "output": Optional[str], "status": "running"|"done"}
        self._activity: list[dict[str, Any]] = []
        self._tool_index: dict[str, int] = {}  # call_id → index in _activity

    # ---- public event hooks (consumed by app.py) -----------------------

    async def on_thought(self, text: str) -> None:
        if not text:
            return
        self._activity.append({"kind": "thought", "text": text})
        await self._update_status()

    async def on_tool_call(self, call_id: str, name: str, args: str) -> None:
        self._activity.append(
            {
                "kind": "tool",
                "call_id": call_id,
                "name": name,
                "args": args or "",
                "output": None,
                "status": "running",
            }
        )
        self._tool_index[call_id] = len(self._activity) - 1
        await self._update_status()

    async def on_tool_output(self, call_id: str, output: str) -> None:
        idx = self._tool_index.get(call_id)
        if idx is not None:
            self._activity[idx]["output"] = output or ""
            self._activity[idx]["status"] = "done"
        else:
            # Result without a matching call — render as standalone "done" entry.
            self._activity.append(
                {
                    "kind": "tool",
                    "call_id": call_id,
                    "name": "tool_result",
                    "args": "",
                    "output": output or "",
                    "status": "done",
                }
            )
        await self._update_status()

    async def on_text_delta(self, token: str) -> None:
        if not token:
            return
        if self.text_msg is None:
            # Create AFTER status so this sits below it in the chat thread.
            self.text_msg = cl.Message(content="")
            await self.text_msg.send()
        await self.text_msg.stream_token(token)

    async def finalize(self) -> None:
        """Flush any pending updates after the event stream completes."""
        if self.text_msg is not None:
            await self.text_msg.update()
        if self.status_msg is not None:
            # Mark all running tools as terminated (defensive — orphan results).
            mutated = False
            for entry in self._activity:
                if entry.get("kind") == "tool" and entry["status"] == "running":
                    entry["status"] = "done"
                    mutated = True
            if mutated:
                await self._update_status()

    # ---- internal rendering -------------------------------------------

    async def _update_status(self) -> None:
        body = "\n\n".join(line for line in (self._render_entry(e) for e in self._activity) if line) or "_Working…_"

        if self.status_msg is None:
            # First call — create with the body in one shot rather than send
            # an empty message and immediately update it (extra WS round-trip,
            # confuses Chainlit's update debouncer).
            self.status_msg = cl.Message(content=body)
            await self.status_msg.send()
        else:
            self.status_msg.content = body
            await self.status_msg.update()

    def _render_entry(self, entry: dict[str, Any]) -> str:
        if entry["kind"] == "thought":
            return self._render_thought(entry["text"])
        if entry["kind"] == "tool":
            return self._render_tool(entry)
        return ""

    def _render_thought(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        if len(text) <= _THOUGHT_INLINE_LIMIT and "\n" not in text:
            return f"💭 _{text}_"
        # Long / multi-line thought → collapsible
        first_line = text.split("\n", 1)[0]
        summary_preview = first_line[:_THOUGHT_INLINE_LIMIT]
        if len(first_line) > _THOUGHT_INLINE_LIMIT:
            summary_preview += "…"
        return f"<details><summary>💭 <i>{html.escape(summary_preview)}</i></summary>\n\n{text}\n\n</details>"

    def _render_tool(self, entry: dict[str, Any]) -> str:
        name = entry["name"]
        args = (entry.get("args") or "").strip()
        output = entry.get("output")
        running = entry["status"] == "running"

        if running:
            head = f"🛠️ <b>{html.escape(name)}</b> running…"
        else:
            head = f"✅ <b>{html.escape(name)}</b> completed"

        sections: list[str] = []
        if _is_meaningful(args):
            sections.append(f"**Args**: `{_escape_inline(args)}`")
        if output is not None and output != "":
            preview = _truncate(output, _RESULT_PREVIEW_LIMIT)
            # Render result as markdown — tools that return structured docs
            # get proper headers/lists/links rendered.
            # The earlier `\`\`\`text` fence both suppressed markdown rendering
            # AND surfaced "text" as a visible language badge (Chainlit's
            # code-block component renders the language name as a header).
            sections.append(f"```\n{preview}\n```")

        if not sections:
            # Nothing to expand — render a flat status line.
            return f"- {head}"

        body = "\n\n".join(sections)
        # Open by default while running so the user sees args without clicking.
        open_attr = " open" if running else ""
        return f"<details{open_attr}><summary>{head}</summary>\n\n{body}\n\n</details>"
