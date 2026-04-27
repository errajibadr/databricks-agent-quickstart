import asyncio
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, AsyncIterator

from mlflow.types.responses import ResponsesAgentRequest


def _load_agent_module(module_spec: str):
    """Load a module by either dotted import name or filesystem path.

    `03_agent.py` cannot be imported with `importlib.import_module("03_agent")`
    because Python identifiers cannot begin with a digit. Path-based loading via
    `importlib.util.spec_from_file_location` sidesteps that constraint.
    """
    if module_spec.endswith(".py") or os.sep in module_spec or "/" in module_spec:
        path = Path(module_spec).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Local agent module not found: {path}")

        sane_name = "_local_agent_" + "".join(
            c if c.isalnum() or c == "_" else "_" for c in path.stem
        )
        spec = importlib.util.spec_from_file_location(sane_name, str(path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not build import spec for {path}")
        module = importlib.util.module_from_spec(spec)

        parent_dir = str(path.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        sys.modules[sane_name] = module
        spec.loader.exec_module(module)
        return module

    return importlib.import_module(module_spec)


class LocalAgentBackend:
    """In-process backend: imports a module exposing `AGENT` and runs `predict_stream`.

    The agent's `predict_stream` is a sync generator that issues blocking LLM /
    Vector Search calls. We pump it in a thread executor so Chainlit's event loop
    stays responsive during the request. Each yielded value is a native
    `ResponsesAgentStreamEvent`, surfaced unchanged.
    """

    def __init__(self, module_path: str):
        self._module_path = module_path
        module = _load_agent_module(module_path)
        if not hasattr(module, "AGENT"):
            raise AttributeError(
                f"{module_path} does not expose a module-level `AGENT` attribute. "
                "ResponsesAgent convention: `AGENT = MyAgent(); mlflow.models.set_model(AGENT)`."
            )
        self.agent = module.AGENT

    async def stream(self, messages: list[dict]) -> AsyncIterator[Any]:
        request = ResponsesAgentRequest(input=messages)
        loop = asyncio.get_running_loop()
        sentinel = object()
        gen = self.agent.predict_stream(request)

        try:
            while True:
                event = await loop.run_in_executor(None, lambda: next(gen, sentinel))
                if event is sentinel:
                    break
                yield event
        finally:
            close = getattr(gen, "close", None)
            if callable(close):
                close()
