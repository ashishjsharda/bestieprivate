"""
agent.py — The BestiePrivate agent loop.

Per user turn:
  1. Slash commands (/remember, /recall, /forget, /goals, ...) run locally with NO model call —
     deterministic, instant, and a reliable demo path.
  2. Otherwise: build a system prompt with auto-recalled memories (keyword + recent), open goals,
     the latest conversation summary, and the skill catalog.
  3. Call Nemotron on Nebius Token Factory with tool schemas; execute any tool calls locally;
     loop until the model answers (max N rounds).
  4. If the selected model/endpoint rejects tool calling, degrade gracefully to plain chat.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

from openai import BadRequestError, OpenAI

import llm
import tools
from memory import MemoryStore

MAX_TOOL_ROUNDS = 5
HISTORY_TURNS = 12  # how many prior messages from this session to send

SYSTEM_TEMPLATE = """You are Bestie, a private personal assistant. You run on an NVIDIA Nemotron model via Nebius Token Factory, \
and the user's memory lives only in a local SQLite file on their own machine.

Today is {today}.

How to behave:
- Be warm, direct, and concise. Use the user's stored context when it genuinely helps; never invent memories.
- When the user shares a durable fact (people, preferences, dates, plans) or says "remember", call save_memory.
- If you need facts not shown below, call recall_memory before saying you don't know.
- Use add_goal / list_goals / complete_goal for goals and to-dos; draft_message for writing messages (never claim to send);
  save_skill / run_skill for reusable workflows; summarize_recent when asked to wrap up or save the conversation.
- You cannot delete memories. If asked, tell the user to use the Memory tab or /forget <id>.
- After a tool runs, confirm briefly in natural language what happened.

=== Relevant memories (auto-recalled from local store) ===
{facts}

=== Open goals ===
{goals}

=== Most recent conversation summary ===
{summary}

=== Saved skills (run with run_skill) ===
{skills}
"""

HELP_TEXT = """**Slash commands** (run locally, no model call):
- `/remember <fact>` — save a memory
- `/recall <keywords>` — search memories
- `/forget <id>` — delete a memory (only you can delete)
- `/goals` — list open goals · `/goal <text>` — add a goal · `/done <id>` — complete a goal
- `/skills` — list saved skills · `/run <skill> [input]` — run a skill
- `/summarize` — summarize this session into memory
- `/help` — this list

Or just talk naturally — Bestie will use its tools on its own."""


@dataclass
class TurnResult:
    reply: str
    tool_events: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    model: str = ""
    used_llm: bool = True
    note: str = ""


def build_system_prompt(store: MemoryStore, user_text: str) -> str:
    relevant = store.search_facts(user_text, limit=8)
    recent = store.list_facts(limit=5)
    seen, facts = set(), []
    for f in relevant + recent:  # relevant first, then most recent, de-duplicated
        if f["id"] not in seen:
            seen.add(f["id"])
            facts.append(f)
    facts_txt = "\n".join(f"- [#{f['id']}] {f['content']}" for f in facts) or "(none yet)"
    goals = store.list_goals("open")[:10]
    goals_txt = "\n".join(
        f"- [{g['id']}] {g['text']}" + (f" (due {g['due']})" if g["due"] else "") for g in goals
    ) or "(none)"
    summaries = store.list_summaries(limit=1)
    summary_txt = summaries[0]["content"] if summaries else "(none)"
    skills_txt = "\n".join(f"- {s['name']}: {s['description']}" for s in store.list_skills()) or "(none)"
    return SYSTEM_TEMPLATE.format(
        today=datetime.now().strftime("%A, %B %d, %Y"),
        facts=facts_txt,
        goals=goals_txt,
        summary=summary_txt,
        skills=skills_txt,
    )


# ----------------------------------------------------------------------------- slash commands

def handle_slash(text: str, ctx: tools.ToolContext) -> TurnResult | None:
    if not text.startswith("/"):
        return None
    cmd, _, arg = text.partition(" ")
    cmd, arg = cmd.lower(), arg.strip()
    store = ctx.store

    def local(reply: str, event: dict | None = None) -> TurnResult:
        return TurnResult(reply=reply, tool_events=[event] if event else [], used_llm=False, model="local")

    if cmd in ("/help", "/?"):
        return local(HELP_TEXT)
    if cmd == "/remember":
        if not arg:
            return local("Usage: `/remember <fact>`")
        res = tools.save_memory(ctx, arg, tags="manual")
        return local(f"🧠 Saved memory **#{res['id']}**: {arg}", {"tool": "save_memory", "args": {"content": arg}, "result": res})
    if cmd == "/recall":
        res = tools.recall_memory(ctx, arg or "", limit=10)
        if not res["results"]:
            return local("No matching memories.", {"tool": "recall_memory", "args": {"query": arg}, "result": res})
        lines = "\n".join(f"- **#{r['id']}** {r['content']}" for r in res["results"])
        return local(f"🔎 Memories matching *{arg or 'recent'}*:\n{lines}", {"tool": "recall_memory", "args": {"query": arg}, "result": res})
    if cmd == "/forget":
        if not arg.lstrip("#").isdigit():
            return local("Usage: `/forget <id>` (ids are shown in the Memory tab and in /recall)")
        ok = store.delete_fact(int(arg.lstrip("#")))
        return local(f"🗑️ Deleted memory #{arg.lstrip('#')}." if ok else f"No memory #{arg}.")
    if cmd == "/goals":
        goals = store.list_goals("open")
        return local("\n".join(f"- **[{g['id']}]** {g['text']}" + (f" _(due {g['due']})_" if g["due"] else "") for g in goals) or "No open goals.")
    if cmd == "/goal":
        if not arg:
            return local("Usage: `/goal <text>`")
        res = tools.add_goal(ctx, arg)
        return local(f"🎯 Added goal **[{res['id']}]** {arg}", {"tool": "add_goal", "args": {"text": arg}, "result": res})
    if cmd == "/done":
        if not arg.isdigit():
            return local("Usage: `/done <goal id>`")
        res = tools.complete_goal(ctx, int(arg))
        return local("✅ Marked done." if res["ok"] else res["error"])
    if cmd == "/skills":
        return local("\n".join(f"- **{s['name']}** — {s['description']} _(run {s['run_count']}×)_" for s in store.list_skills()) or "No skills saved.")
    if cmd == "/run":
        name, _, inp = arg.partition(" ")
        if not name:
            return local("Usage: `/run <skill-name> [input]`")
        res = tools.run_skill(ctx, name, inp)
        return TurnResult(reply=res.get("output") or res.get("error", "Skill failed."),
                          tool_events=[{"tool": "run_skill", "args": {"name": name, "input": inp}, "result": {k: v for k, v in res.items() if k != "output"}}],
                          used_llm=True, model=ctx.model)
    if cmd == "/summarize":
        res = tools.summarize_recent(ctx)
        return TurnResult(reply=("📝 Saved summary:\n\n" + res["summary"]) if res["ok"] else res["error"],
                          tool_events=[{"tool": "summarize_recent", "args": {}, "result": {"ok": res["ok"]}}],
                          used_llm=res["ok"], model=ctx.model)
    return local(f"Unknown command `{cmd}`.\n\n{HELP_TEXT}")


# ----------------------------------------------------------------------------- main loop

def _assistant_msg_dict(msg) -> dict:
    """Convert an SDK message with tool_calls into a plain dict we can send back."""
    return {
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}}
            for tc in msg.tool_calls
        ],
    }


def run_turn(
    store: MemoryStore,
    client: OpenAI | None,
    model: str,
    session_id: str,
    user_text: str,
    use_tools: bool = True,
    persist: bool = True,
) -> TurnResult:
    """Process one user message end-to-end. `persist=False` = Incognito (transcript not stored)."""
    stats = llm.CallStats()
    ctx = tools.ToolContext(store=store, client=client, model=model, session_id=session_id, stats=stats)

    # History is read BEFORE we store the new message so it isn't duplicated.
    history = store.get_messages(session_id, limit=HISTORY_TURNS) if persist else []
    if persist:
        store.add_message(session_id, "user", user_text)

    result = handle_slash(user_text.strip(), ctx)
    if result is None:
        if client is None:
            raise llm.MissingAPIKeyError("NEBIUS_API_KEY is not set.")
        result = _llm_turn(ctx, user_text, history, use_tools)

    result.stats = stats.as_dict()
    if persist:
        store.add_message(session_id, "assistant", result.reply,
                          meta={"tool_events": result.tool_events, "stats": result.stats, "model": result.model, "note": result.note})
    return result


def _llm_turn(ctx: tools.ToolContext, user_text: str, history: list[dict], use_tools: bool) -> TurnResult:
    messages: list[dict] = [{"role": "system", "content": build_system_prompt(ctx.store, user_text)}]
    messages += [{"role": m["role"], "content": m["content"]} for m in history if m["role"] in ("user", "assistant")]
    messages.append({"role": "user", "content": user_text})

    events: list[dict] = []
    note = ""
    schemas = tools.TOOL_SCHEMAS if use_tools else None

    for _ in range(MAX_TOOL_ROUNDS):
        try:
            msg = llm.chat(ctx.client, ctx.model, messages, tools=schemas, stats=ctx.stats)
        except BadRequestError as e:
            if schemas is None:
                raise
            # Endpoint/model refused tool calling -> fall back to plain chat so the demo never dies.
            note = f"Tool calling unavailable for {ctx.model}; answered without tools. ({str(e)[:120]})"
            schemas = None
            continue

        if not getattr(msg, "tool_calls", None):
            reply = llm.strip_reasoning(msg.content) or "(no response)"
            return TurnResult(reply=reply, tool_events=events, model=ctx.model, note=note)

        messages.append(_assistant_msg_dict(msg))
        for tc in msg.tool_calls:
            res = tools.execute(ctx, tc.function.name, tc.function.arguments)
            try:
                parsed_args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                parsed_args = {"raw": tc.function.arguments}
            events.append({"tool": tc.function.name, "args": parsed_args, "result": res})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(res, default=str)[:6000]})

    # Ran out of rounds: ask once more for a final answer without tools.
    msg = llm.chat(ctx.client, ctx.model, messages, tools=None, stats=ctx.stats)
    return TurnResult(reply=llm.strip_reasoning(msg.content) or "(no response)", tool_events=events,
                      model=ctx.model, note=note or "Reached tool-call limit.")
