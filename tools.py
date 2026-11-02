"""
tools.py — Reusable skills/tools the Nemotron agent can call via OpenAI-style tool calling.

Each tool = a JSON schema (what the model sees) + a Python function (what actually runs,
locally, against your SQLite memory). Deleting memories is intentionally NOT exposed to the
model — only you can delete, from the UI or with /forget. The model can add, never erase.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from openai import OpenAI

import llm
from memory import MemoryStore


@dataclass
class ToolContext:
    store: MemoryStore
    client: OpenAI | None
    model: str
    session_id: str
    stats: llm.CallStats


def _fn(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required or []},
        },
    }


TOOL_SCHEMAS: list[dict] = [
    _fn(
        "save_memory",
        "Save a durable fact about the user to their private local memory. Use when the user shares a lasting "
        "preference, person, plan, or detail, or explicitly asks you to remember something. One fact per call, "
        "written in third person (e.g. 'User's sister is Priya; she lives in Austin').",
        {
            "content": {"type": "string", "description": "The fact, short and self-contained."},
            "tags": {"type": "string", "description": "Optional comma-separated tags, e.g. 'family,birthday'."},
        },
        ["content"],
    ),
    _fn(
        "recall_memory",
        "Search the user's private memory for facts relevant to a query. Use before answering questions about "
        "the user's life, people, preferences, or past plans if the facts are not already in context.",
        {
            "query": {"type": "string", "description": "Keywords to search for."},
            "limit": {"type": "integer", "description": "Max results (default 5)."},
        },
        ["query"],
    ),
    _fn(
        "summarize_recent",
        "Summarize the recent conversation into a short note and store it in memory, so future sessions keep "
        "the context. Use when the user asks to summarize, wrap up, or save this conversation.",
        {"n_messages": {"type": "integer", "description": "How many recent messages to summarize (default 20)."}},
    ),
    _fn(
        "add_goal",
        "Add a goal or to-do item to the user's goal list.",
        {
            "text": {"type": "string", "description": "The goal."},
            "due": {"type": "string", "description": "Optional due date/time in plain words or ISO format."},
        },
        ["text"],
    ),
    _fn(
        "list_goals",
        "List the user's goals.",
        {"status": {"type": "string", "enum": ["open", "done", "all"], "description": "Default 'open'."}},
    ),
    _fn(
        "complete_goal",
        "Mark a goal as done by its numeric id.",
        {"goal_id": {"type": "integer"}},
        ["goal_id"],
    ),
    _fn(
        "draft_message",
        "Draft an email, text, or Slack message on the user's behalf, personalized with what's in their memory. "
        "Returns the draft only — it is never sent.",
        {
            "recipient": {"type": "string", "description": "Who it's for (name or relationship)."},
            "purpose": {"type": "string", "description": "What the message should accomplish."},
            "tone": {"type": "string", "description": "e.g. warm, formal, brief. Default warm."},
            "channel": {"type": "string", "enum": ["email", "text", "slack"], "description": "Default text."},
        },
        ["recipient", "purpose"],
    ),
    _fn(
        "save_skill",
        "Create or update a reusable skill: a named workflow (prompt instructions) the user wants to re-run later, "
        "e.g. 'gift-ideas' or 'meeting-prep'. Use '{input}' in the instructions where the user's input goes.",
        {
            "name": {"type": "string", "description": "short-kebab-case name"},
            "description": {"type": "string", "description": "One line: what the skill does."},
            "instructions": {"type": "string", "description": "Full instructions for running the skill."},
        },
        ["name", "instructions"],
    ),
    _fn(
        "run_skill",
        "Run one of the user's saved skills by name with optional input. Check the list of skills in context.",
        {
            "name": {"type": "string"},
            "input": {"type": "string", "description": "Input for the skill (optional)."},
        },
        ["name"],
    ),
]


# --------------------------------------------------------------------------- implementations

def _user_context(store: MemoryStore, query: str = "", k: int = 8) -> str:
    facts = store.search_facts(query, limit=k) if query else store.list_facts(limit=k)
    goals = store.list_goals("open")[:8]
    lines = ["Known facts about the user:"] + [f"- {f['content']}" for f in facts] if facts else []
    if goals:
        lines += ["Open goals:"] + [f"- [{g['id']}] {g['text']}" + (f" (due {g['due']})" if g["due"] else "") for g in goals]
    return "\n".join(lines) or "(no stored context yet)"


def save_memory(ctx: ToolContext, content: str, tags: str = "") -> dict:
    fid = ctx.store.add_fact(content, tags=tags, source="agent")
    return {"ok": True, "id": fid, "saved": content}


def recall_memory(ctx: ToolContext, query: str, limit: int = 5) -> dict:
    hits = ctx.store.search_facts(query, limit=max(1, min(int(limit or 5), 20)))
    return {"ok": True, "results": [{"id": h["id"], "content": h["content"], "tags": h["tags"]} for h in hits]}


def summarize_recent(ctx: ToolContext, n_messages: int = 20) -> dict:
    msgs = ctx.store.get_messages(ctx.session_id, limit=max(2, min(int(n_messages or 20), 100)))
    if not msgs:
        return {"ok": False, "error": "Nothing to summarize yet in this session (or Incognito mode is on)."}
    if ctx.client is None:
        return {"ok": False, "error": "No model client available."}
    transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in msgs)
    summary = llm.complete(
        ctx.client,
        ctx.model,
        "You compress chat transcripts into durable notes for a personal assistant's memory. "
        "Write 3-6 bullet points: decisions, commitments, facts learned about the user, open questions. "
        "No preamble.",
        transcript[-12000:],
        stats=ctx.stats,
        max_tokens=500,
    )
    sid = ctx.store.add_summary(summary, session_id=ctx.session_id)
    return {"ok": True, "id": sid, "summary": summary}


def add_goal(ctx: ToolContext, text: str, due: str = "") -> dict:
    gid = ctx.store.add_goal(text, due=due or "")
    return {"ok": True, "id": gid, "goal": text, "due": due}


def list_goals(ctx: ToolContext, status: str = "open") -> dict:
    status = status if status in ("open", "done", "all") else "open"
    goals = ctx.store.list_goals(status)
    return {"ok": True, "goals": [{"id": g["id"], "text": g["text"], "due": g["due"], "status": g["status"]} for g in goals]}


def complete_goal(ctx: ToolContext, goal_id: int) -> dict:
    ok = ctx.store.set_goal_status(int(goal_id), "done")
    return {"ok": ok, "goal_id": goal_id} if ok else {"ok": False, "error": f"No goal with id {goal_id}."}


def draft_message(ctx: ToolContext, recipient: str, purpose: str, tone: str = "warm", channel: str = "text") -> dict:
    if ctx.client is None:
        return {"ok": False, "error": "No model client available."}
    context = _user_context(ctx.store, query=f"{recipient} {purpose}")
    draft = llm.complete(
        ctx.client,
        ctx.model,
        "You ghost-write messages for the user in their own voice. Use relevant facts only when natural; "
        "never invent facts. Output ONLY the message (for email, include a 'Subject:' line first).",
        f"{context}\n\nWrite a {tone} {channel} to {recipient}. Purpose: {purpose}",
        stats=ctx.stats,
        max_tokens=500,
        temperature=0.7,
    )
    return {"ok": True, "draft": draft, "note": "Draft only — not sent."}


def save_skill(ctx: ToolContext, name: str, instructions: str, description: str = "") -> dict:
    key = ctx.store.upsert_skill(name, description, instructions)
    return {"ok": True, "skill": key}


def run_skill(ctx: ToolContext, name: str, input: str = "") -> dict:  # noqa: A002 (matches schema)
    skill = ctx.store.get_skill(name)
    if not skill:
        return {"ok": False, "error": f"No skill named '{name}'.",
                "available": [s["name"] for s in ctx.store.list_skills()]}
    if ctx.client is None:
        return {"ok": False, "error": "No model client available."}
    instr = skill["instructions"]
    instr = instr.replace("{input}", input or "(none)") if "{input}" in instr else f"{instr}\n\nInput: {input or '(none)'}"
    out = llm.complete(
        ctx.client,
        ctx.model,
        "You are executing a saved personal skill for the user. Follow the instructions exactly.",
        f"{_user_context(ctx.store, query=input)}\n\n--- SKILL: {skill['name']} ---\n{instr}",
        stats=ctx.stats,
        max_tokens=900,
    )
    ctx.store.bump_skill(skill["name"])
    return {"ok": True, "skill": skill["name"], "output": out}


REGISTRY: dict[str, Callable[..., dict]] = {
    "save_memory": save_memory,
    "recall_memory": recall_memory,
    "summarize_recent": summarize_recent,
    "add_goal": add_goal,
    "list_goals": list_goals,
    "complete_goal": complete_goal,
    "draft_message": draft_message,
    "save_skill": save_skill,
    "run_skill": run_skill,
}


def execute(ctx: ToolContext, name: str, raw_args: str | dict | None) -> dict:
    """Run a tool safely. Bad JSON, unknown tools, or bad args become error results the model can read."""
    fn = REGISTRY.get(name)
    if fn is None:
        return {"ok": False, "error": f"Unknown tool '{name}'."}
    try:
        args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else json.loads(raw_args or "{}")
    except json.JSONDecodeError as e:
        return {"ok": False, "error": f"Arguments were not valid JSON: {e}"}
    try:
        return fn(ctx, **args)
    except TypeError as e:
        return {"ok": False, "error": f"Bad arguments for {name}: {e}"}
    except Exception as e:  # keep the chat alive no matter what
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
