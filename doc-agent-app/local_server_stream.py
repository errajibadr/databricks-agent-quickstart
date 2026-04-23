"""Test streaming from the local agent server using OpenAI client."""

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8181", api_key="local")

print("--- Streaming ---\n")
stream = client.responses.create(
    model="agent",
    input=[{"role": "user", "content": "hello, tell me a joke"}],
    stream=True,
)
for event in stream:
    etype = event.type
    if etype == "response.output_text.delta":
        print(event.delta, end="|", flush=True)
    else:
        print(f"\n[{etype}]")

print("\n\n--- Done ---")
