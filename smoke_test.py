"""
smoke_test.py — 30-second check that your Nebius key, endpoint and Nemotron model IDs work.

    python smoke_test.py            # lists NVIDIA models + one chat + one tool call on the default model
    python smoke_test.py <model-id> # test a specific model
"""

import json
import sys
import time

import llm
import tools


def main() -> int:
    try:
        client = llm.get_client()
    except llm.MissingAPIKeyError as e:
        print(f"❌ {e}")
        return 1

    print("→ Listing models on Nebius Token Factory…")
    try:
        ids = sorted(m.id for m in client.models.list().data)
        nv = [i for i in ids if "nemotron" in i.lower() or i.lower().startswith("nvidia/")]
        print(f"✅ {len(ids)} models available. NVIDIA/Nemotron ones:")
        for i in nv:
            mark = "  ← configured" if i in llm.MODELS.values() else ""
            print(f"   - {i}{mark}")
        missing = [m for m in llm.MODELS.values() if m not in ids]
        if missing:
            print(f"⚠️  Configured but not listed (check spelling / access): {missing}")
    except Exception as e:
        print(f"⚠️  Could not list models ({type(e).__name__}: {e}). Continuing.")

    model = sys.argv[1] if len(sys.argv) > 1 else llm.DEFAULT_MODEL
    print(f"\n→ Chat test on {model}…")
    t0 = time.perf_counter()
    msg = llm.chat(client, model, [{"role": "user", "content": "Reply with exactly: bestie online"}], max_tokens=64)
    print(f"✅ {llm.strip_reasoning(msg.content)!r} in {time.perf_counter() - t0:.2f}s")

    print(f"\n→ Tool-calling test on {model}…")
    msg = llm.chat(
        client, model,
        [{"role": "system", "content": "Use tools when appropriate."},
         {"role": "user", "content": "Remember that my favorite color is teal."}],
        tools=tools.TOOL_SCHEMAS, max_tokens=256,
    )
    if getattr(msg, "tool_calls", None):
        tc = msg.tool_calls[0]
        print(f"✅ Model called {tc.function.name}({json.dumps(json.loads(tc.function.arguments or '{}'))})")
    else:
        print(f"⚠️  No tool call returned; reply was: {llm.strip_reasoning(msg.content)[:200]!r}")
        print("   The app still works (slash commands + fallback), but try another model for the agent demo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
