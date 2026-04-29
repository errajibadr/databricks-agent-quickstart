"""
Query a deployed Databricks agent serving endpoint.

Endpoint:
  https://adb-xxxxx-xxxxx.x.azuredatabricks.net/serving-endpoints/lg-doc-agent/invocations

Auth: uses Databricks CLI profile (DATABRICKS_CONFIG_PROFILE env var, default "DEFAULT").
      Falls back to DATABRICKS_HOST + DATABRICKS_TOKEN env vars if set.

Run:
  python query_deployed_agent.py            # non-streaming (single response)
  python query_deployed_agent.py --stream   # streaming (observe tool_call / tool_result / tokens)
"""

import argparse
import os

from databricks.sdk import WorkspaceClient
from databricks_openai import DatabricksOpenAI

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ENDPOINT_NAME = "lg-doc-agent"
QUERY = "how many doc do you have"


def run_non_streaming(client: DatabricksOpenAI) -> None:
    resp = client.responses.create(
        model=ENDPOINT_NAME,
        input=[{"role": "user", "content": QUERY}],
    )
    for item in resp.output:
        itype = item.type
        if itype == "message":
            for block in item.content or []:
                text = getattr(block, "text", "")
                if text:
                    print(text)
        elif itype == "function_call":
            print(f"[TOOL_CALL] {item.name}({item.arguments})")
        elif itype == "function_call_output":
            preview = (item.output or "")[:300].replace("\n", " ")
            print(f"[TOOL_RESULT] {preview}...")
        elif itype == "reasoning":
            print("[REASONING] (present)")


def run_streaming(client: DatabricksOpenAI) -> None:
    stream = client.responses.create(
        model=ENDPOINT_NAME,
        input=[{"role": "user", "content": QUERY}],
        stream=True,
    )
    for event in stream:
        import time

        print(time.ctime())

        etype = getattr(event, "type", "?")

        if etype == "response.output_text.delta":
            print(getattr(event, "delta", ""), end="", flush=True)

        elif etype == "response.output_item.done":
            item = getattr(event, "item", None)
            if item is None:
                continue
            itype = getattr(item, "type", None)
            if itype == "function_call":
                print(f"\n[TOOL_CALL] {item.name}({item.arguments})")
            elif itype == "function_call_output":
                preview = (getattr(item, "output", "") or "")[:300].replace("\n", " ")
                print(f"\n[TOOL_RESULT] {preview}...")

    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--not-stream", action="store_true", help="Stream events instead of single response")
    args = parser.parse_args()

    profile = os.environ.get("DATABRICKS_CONFIG_PROFILE", "DEFAULT")
    w = WorkspaceClient(profile=profile)
    client = DatabricksOpenAI(workspace_client=w)

    print(f"Workspace: {w.config.host}")
    print(f"Endpoint:  {ENDPOINT_NAME}")
    print(f"Query:     {QUERY!r}")
    print(f"Mode:      {'streaming' if not args.not_stream else 'single-shot'}")
    print("-" * 60)

    if not args.not_stream:
        run_streaming(client)
    else:
        run_non_streaming(client)


if __name__ == "__main__":
    main()
