"""Test streaming from the local agent server using OpenAI client.

Prereq: `uv run python start_server.py --port 8181` in another terminal.
"""

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8181", api_key="local")

print("--- Streaming ---\n")
stream = client.responses.create(
    model="agent",
    input=[{"role": "user", "content": "what about the number of documents in the knowledge base?"}],
    stream=True,
)
for event in stream:
    etype = event.type
    if etype == "response.output_text.delta":
        print(event.delta, end="|", flush=True)
    elif etype == "response.output_item.done":
        item = event.item
        itype = getattr(item, "type", None)
        if itype == "function_call":
            print(f"\n🔧 TOOL_CALL     {item.name}({item.arguments})")
        elif itype == "function_call_output":
            preview = (getattr(item, "output", "") or "")[:100].replace("\n", " ")
            print(f"\n✅ TOOL_RESULT   {preview}...")
        elif itype == "message":
            print("\n📝 MESSAGE_DONE  (final text assembled)")
        else:
            print(f"\n[done type={itype}]")
    else:
        print(f"\n[{etype}]")

print("\n\n--- Done ---")
