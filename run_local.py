"""Run 03_agent.py locally for development and debugging.

Usage:
    cp .env.example .env          # then fill in VS_INDEX + Databricks auth
    uv sync                       # install deps from pyproject.toml
    uv run python run_local.py "What is tool calling in LangChain?"
    uv run python run_local.py "What's the weather in Paris?" --stream

The local Python process talks to Databricks-hosted resources (Vector Search
index, LLM endpoint) over HTTPS. Your laptop never holds the index or the
model weights — it just runs the agent's orchestration code.
"""

import argparse
from importlib import import_module

from dotenv import load_dotenv

# Load .env BEFORE importing 03_agent — the agent reads os.environ at import time
# and will raise RuntimeError if VS_INDEX isn't set yet.
load_dotenv()

from mlflow.types.responses import ResponsesAgentRequest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the LangGraph doc agent locally."
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="What is tool calling in LangChain?",
        help="The user message to send to the agent.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream tokens + tool events instead of waiting for full response.",
    )
    args = parser.parse_args()

    # Lazy import so any env-var error from 03_agent surfaces inside main(),
    # not at module load — keeps argparse's --help working without a real .env.
    agent_mod = import_module("03_agent")
    AGENT = agent_mod.AGENT

    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": args.question}]
    )

    if args.stream:
        for event in AGENT.predict_stream(request):
            print(f"[{event.type}]", event)
    else:
        result = AGENT.predict(request)
        for item in result.output:
            print(item)


if __name__ == "__main__":
    main()
