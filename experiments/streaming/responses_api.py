"""
Test: Does DatabricksOpenAI responses.create() work against
Foundation Model API endpoints?

Run: .venv/bin/python test_responses_api.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

from databricks_openai import DatabricksOpenAI
from databricks.sdk import WorkspaceClient

PROFILE = os.environ.get("DATABRICKS_CONFIG_PROFILE", "DEFAULT")
w = WorkspaceClient(profile=PROFILE)
client = DatabricksOpenAI(workspace_client=w)

PROMPT = "Say hello in one sentence."

MODELS = [
    "databricks-gpt-oss-120b",
    "databricks-meta-llama-3-3-70b-instruct",
]

for model in MODELS:
    print(f"\n{'=' * 60}")
    print(f"Model: {model}")
    print(f"{'=' * 60}")

    # Test 1: responses.create (Responses API)
    print("\n  [Responses API] responses.create()...")
    try:
        response = client.responses.create(
            model=model,
            input=[{"role": "user", "content": PROMPT}],
        )
        for item in response.output:
            if item.type == "message":
                for content in item.content or []:
                    print(f"    OK: {getattr(content, 'text', '')[:100]}")
            elif item.type == "reasoning":
                print("    Reasoning: (present)")
    except Exception as e:
        print(f"    FAILED: {e}")

    # Test 2: chat.completions.create (Chat Completions API)
    print("\n  [Chat Completions] chat.completions.create()...")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
        )
        content = response.choices[0].message.content
        if isinstance(content, str):
            print(f"    OK (str): {content[:100]}")
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    print(f"    OK (block type={block.get('type')}): {str(block)[:80]}")
                else:
                    print(f"    OK: {str(block)[:80]}")
        else:
            print(f"    OK ({type(content).__name__}): {str(content)[:100]}")
    except Exception as e:
        print(f"    FAILED: {e}")
