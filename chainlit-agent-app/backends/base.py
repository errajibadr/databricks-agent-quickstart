from typing import Any, AsyncIterator, Protocol


class Backend(Protocol):
    """Yields Responses-API-native stream events.

    Both backends produce events that conform to the Responses API event taxonomy
    (`response.output_text.delta`, `response.output_item.done`, `task_continue_request`,
    etc.). The Chainlit handler in `app.py` pattern-matches on `event.type` directly —
    no custom dataclass layer.

    Concrete event types yielded:
      - LocalAgentBackend  : mlflow.types.responses.ResponsesAgentStreamEvent
      - EndpointBackend    : openai.types.responses.ResponseStreamEvent  (Step B)

    Both expose `.type` plus type-specific fields (`.delta`, `.item`, `.item.type`,
    `.item.call_id`, `.item.name`, `.item.arguments`, `.item.output`, etc.). The
    handler reads them via `getattr(...)` so attribute-vs-dict shape variation
    is absorbed without a dedicated shim.
    """

    async def stream(self, messages: list[dict]) -> AsyncIterator[Any]: ...
