"""
app.py — BestiePrivate Streamlit UI.

Run:  streamlit run app.py
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime

import streamlit as st

import agent
import llm
from memory import MemoryStore

st.set_page_config(page_title="BestiePrivate", page_icon="🫶", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem;}
      .bestie-tool {font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.82rem;
                    padding: 2px 8px; border-radius: 6px; background: rgba(118,185,0,0.14);
                    border: 1px solid rgba(118,185,0,0.45); display: inline-block; margin: 2px 4px 2px 0;}
      .bestie-meta {color: #8a8f98; font-size: 0.78rem;}
      .bestie-pill {font-size: 0.75rem; padding: 2px 8px; border-radius: 999px; border: 1px solid #8a8f98;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------- state
@st.cache_resource
def get_store() -> MemoryStore:
    return MemoryStore()


store = get_store()

if "session_id" not in st.session_state:
    # Every app launch starts a NEW session: memories persist, the chat transcript doesn't carry over.
    st.session_state.session_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
if "incognito_log" not in st.session_state:
    st.session_state.incognito_log = []  # in-RAM transcript when Incognito is on


def new_session() -> None:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    st.session_state.incognito_log = []


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 🫶 BestiePrivate")
    st.caption("Your private AI bestie. Memory stays on this machine.")

    labels = list(llm.MODELS.keys())
    default_idx = next((i for i, l in enumerate(labels) if llm.MODELS[l] == llm.DEFAULT_MODEL), 0)
    model_label = st.selectbox("Nemotron model (Nebius Token Factory)", labels, index=default_idx)
    model_id = llm.MODELS[model_label]
    st.code(model_id, language=None)

    use_tools = st.toggle("Agent tools (tool calling)", value=True,
                          help="Let Nemotron call save_memory, recall_memory, goals, drafts and skills.")
    incognito = st.toggle("🕶️ Incognito (don't store this chat)", value=False,
                          help="Transcript is kept only in browser memory. Memories are saved only if you or Bestie explicitly save one.")

    if llm.api_key_present():
        st.success("Nebius API key loaded", icon="🔑")
    else:
        st.error("NEBIUS_API_KEY missing — add it to .env and restart.", icon="⚠️")

    st.divider()
    s = store.stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("Memories", s["facts"])
    c2.metric("Goals", len(store.list_goals("open")))
    c3.metric("Skills", s["skills"])

    st.divider()
    st.markdown("**Session**")
    st.caption(f"`{st.session_state.session_id}`")
    if st.button("🔄 New session (simulate restart)", use_container_width=True):
        new_session()
        st.rerun()

    sessions = [x for x in store.list_sessions() if x["session_id"] != st.session_state.session_id][:8]
    if sessions:
        with st.expander("Past sessions"):
            for sess in sessions:
                cols = st.columns([3, 1])
                if cols[0].button(f"{sess['session_id']} · {sess['n']} msgs", key=f"open_{sess['session_id']}"):
                    st.session_state.session_id = sess["session_id"]
                    st.rerun()
                if cols[1].button("🗑️", key=f"del_{sess['session_id']}", help="Delete this transcript"):
                    store.delete_session(sess["session_id"])
                    st.rerun()


def get_client():
    try:
        return llm.get_client()
    except llm.MissingAPIKeyError:
        return None


# ----------------------------------------------------------------------------- helpers
def render_tool_events(events: list[dict]) -> None:
    if not events:
        return
    chips = " ".join(f"<span class='bestie-tool'>🔧 {e['tool']}</span>" for e in events)
    st.markdown(chips, unsafe_allow_html=True)
    with st.expander("Tool calls (what ran locally)", expanded=False):
        for e in events:
            st.markdown(f"**{e['tool']}**")
            st.json({"args": e.get("args"), "result": e.get("result")}, expanded=False)


def render_meta(meta: dict) -> None:
    stats = meta.get("stats") or {}
    if not stats or not stats.get("calls"):
        if meta.get("model") == "local":
            st.markdown("<span class='bestie-meta'>⚡ local command · no model call · nothing left your machine</span>", unsafe_allow_html=True)
        return
    cost = stats.get("est_cost_usd")
    cost_txt = f" · ~${cost:.5f}" if cost is not None else ""
    st.markdown(
        f"<span class='bestie-meta'>🟢 {', '.join(stats.get('models', []))} · {stats['calls']} call(s) · "
        f"{stats['latency_s']}s · {stats['prompt_tokens']}→{stats['completion_tokens']} tok{cost_txt}</span>",
        unsafe_allow_html=True,
    )
    if meta.get("note"):
        st.caption(f"ℹ️ {meta['note']}")


# ----------------------------------------------------------------------------- tabs
tab_chat, tab_mem, tab_skills, tab_priv = st.tabs(["💬 Chat", "🧠 Memory", "🛠️ Skills & Goals", "🔒 Privacy"])

# ============================== CHAT ==============================
with tab_chat:
    if incognito:
        history = st.session_state.incognito_log
        st.info("Incognito: this conversation isn't written to disk.", icon="🕶️")
    else:
        history = [
            {"role": m["role"], "content": m["content"], "meta": m.get("meta") or {}}
            for m in store.get_messages(st.session_state.session_id)
        ]

    if not history:
        st.markdown("### Hey bestie 👋")
        st.markdown(
            "I remember what you tell me — **only on this computer**. Try:\n"
            "- *“Remember my sister Priya's birthday is March 3 and she loves orchids.”*\n"
            "- *“Add a goal: finish the hackathon video by Thursday 9am.”*\n"
            "- *“Draft a text to Priya wishing her happy birthday.”*\n"
            "- Then click **New session** in the sidebar and ask *“What do you know about my sister?”*\n\n"
            "Type `/help` for instant local commands."
        )

    for m in history:
        with st.chat_message(m["role"], avatar="🫶" if m["role"] == "assistant" else None):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                render_tool_events((m.get("meta") or {}).get("tool_events") or [])
                render_meta(m.get("meta") or {})

    prompt = st.chat_input("Message your bestie…  (try /help)")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="🫶"):
            client = get_client()
            if client is None and not prompt.strip().startswith("/"):
                st.error("Set NEBIUS_API_KEY in `.env` (see README) and restart. Slash commands like `/remember` still work offline.")
                st.stop()
            with st.spinner(f"Thinking with {model_id}…"):
                try:
                    res = agent.run_turn(store, client, model_id, st.session_state.session_id, prompt,
                                         use_tools=use_tools, persist=not incognito)
                except llm.MissingAPIKeyError as e:
                    st.error(str(e))
                    st.stop()
                except Exception as e:  # network / auth / model errors -> friendly message
                    msg = str(e)
                    hint = ""
                    if "401" in msg or "auth" in msg.lower():
                        hint = " Check that NEBIUS_API_KEY is valid."
                    elif "404" in msg or "not found" in msg.lower():
                        hint = " The model ID may have changed — run `python smoke_test.py` to list available models."
                    st.error(f"Model call failed: {type(e).__name__}: {msg[:300]}{hint}")
                    st.stop()
            st.markdown(res.reply)
            meta = {"tool_events": res.tool_events, "stats": res.stats, "model": res.model, "note": res.note}
            render_tool_events(res.tool_events)
            render_meta(meta)
        if incognito:
            st.session_state.incognito_log += [
                {"role": "user", "content": prompt, "meta": {}},
                {"role": "assistant", "content": res.reply, "meta": meta},
            ]
        if any(e["tool"] in ("save_memory", "add_goal", "save_skill", "complete_goal") for e in res.tool_events):
            st.toast("Memory updated — see the 🧠 Memory tab", icon="🧠")
        st.rerun()

# ============================== MEMORY ==============================
with tab_mem:
    st.markdown("### 🧠 Everything Bestie remembers")
    st.caption(f"Stored locally at `{os.path.abspath(store.db_path)}`. Edit or delete anything — you're in control.")

    with st.form("add_fact", clear_on_submit=True):
        c1, c2, c3 = st.columns([6, 2, 1])
        new_fact = c1.text_input("Add a memory", placeholder="e.g. I'm vegetarian and allergic to cashews")
        new_tags = c2.text_input("Tags", placeholder="food,health")
        if c3.form_submit_button("Save", use_container_width=True) and new_fact.strip():
            store.add_fact(new_fact, tags=new_tags, source="manual")
            st.rerun()

    q = st.text_input("🔎 Search memories", placeholder="keywords…")
    facts = store.search_facts(q, limit=200) if q else store.list_facts()
    if not facts:
        st.info("No memories yet. Tell Bestie something about yourself in Chat.")
    for f in facts:
        with st.container(border=True):
            c1, c2, c3 = st.columns([7, 2, 2])
            edited = c1.text_area(f"#{f['id']}", f["content"], key=f"fact_{f['id']}", height=68, label_visibility="collapsed")
            tags = c2.text_input("tags", f["tags"] or "", key=f"tags_{f['id']}", label_visibility="collapsed", placeholder="tags")
            c2.caption(f"#{f['id']} · {f['source']} · {f['updated_at'][:16].replace('T', ' ')}")
            if c3.button("💾 Save", key=f"save_{f['id']}", use_container_width=True):
                store.update_fact(f["id"], edited, tags)
                st.toast(f"Updated #{f['id']}")
                st.rerun()
            if c3.button("🗑️ Delete", key=f"delf_{f['id']}", use_container_width=True):
                store.delete_fact(f["id"])
                st.rerun()

    st.markdown("### 📝 Conversation summaries")
    sums = store.list_summaries()
    if not sums:
        st.caption("None yet — ask Bestie to “summarize our chat” or type `/summarize`.")
    for sm in sums:
        with st.expander(f"Summary #{sm['id']} · {sm['created_at'][:16].replace('T', ' ')}"):
            txt = st.text_area("summary", sm["content"], key=f"sum_{sm['id']}", height=140, label_visibility="collapsed")
            c1, c2 = st.columns(2)
            if c1.button("💾 Save", key=f"savesum_{sm['id']}"):
                store.update_summary(sm["id"], txt)
                st.rerun()
            if c2.button("🗑️ Delete", key=f"delsum_{sm['id']}"):
                store.delete_summary(sm["id"])
                st.rerun()

# ============================== SKILLS & GOALS ==============================
with tab_skills:
    left, right = st.columns(2)
    with left:
        st.markdown("### 🛠️ Reusable skills")
        st.caption("Named workflows Bestie can re-run. Use `{input}` where your input goes. Run with “run my daily-plan skill” or `/run daily-plan`.")
        for sk in store.list_skills():
            with st.expander(f"**{sk['name']}** — {sk['description']}  ·  run {sk['run_count']}×"):
                d = st.text_input("Description", sk["description"], key=f"skd_{sk['name']}")
                ins = st.text_area("Instructions", sk["instructions"], key=f"ski_{sk['name']}", height=150)
                c1, c2 = st.columns(2)
                if c1.button("💾 Save", key=f"sks_{sk['name']}"):
                    store.upsert_skill(sk["name"], d, ins)
                    st.rerun()
                if c2.button("🗑️ Delete", key=f"skx_{sk['name']}"):
                    store.delete_skill(sk["name"])
                    st.rerun()
        with st.form("new_skill", clear_on_submit=True):
            st.markdown("**New skill**")
            n = st.text_input("Name", placeholder="gift-ideas")
            d = st.text_input("Description", placeholder="Suggest 5 gifts for someone using what you know about them")
            i = st.text_area("Instructions", placeholder="Suggest 5 thoughtful gift ideas under $50 for: {input}. Use known facts about that person.")
            if st.form_submit_button("Create skill") and n and i:
                try:
                    store.upsert_skill(n, d, i)
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))

    with right:
        st.markdown("### 🎯 Goals")
        with st.form("new_goal", clear_on_submit=True):
            c1, c2, c3 = st.columns([5, 2, 1])
            gt = c1.text_input("Goal", label_visibility="collapsed", placeholder="Add a goal…")
            gd = c2.text_input("Due", label_visibility="collapsed", placeholder="due (optional)")
            if c3.form_submit_button("Add") and gt.strip():
                store.add_goal(gt, gd)
                st.rerun()
        for g in store.list_goals("all"):
            c1, c2, c3 = st.columns([7, 1, 1])
            done = g["status"] == "done"
            label = f"~~{g['text']}~~" if done else g["text"]
            c1.markdown(f"**[{g['id']}]** {label}" + (f"  \n<span class='bestie-meta'>due {g['due']}</span>" if g["due"] else ""),
                        unsafe_allow_html=True)
            if c2.button("↩️" if done else "✅", key=f"g_{g['id']}", help="Toggle done"):
                store.set_goal_status(g["id"], "open" if done else "done")
                st.rerun()
            if c3.button("🗑️", key=f"gx_{g['id']}"):
                store.delete_goal(g["id"])
                st.rerun()

# ============================== PRIVACY ==============================
with tab_priv:
    st.markdown("### 🔒 Your data, your rules")
    st.markdown(
        f"""
- **Where memory lives:** one SQLite file on this machine → `{os.path.abspath(store.db_path)}`
- **What leaves your machine:** only the model request to Nebius Token Factory (`{os.getenv('NEBIUS_BASE_URL', llm.DEFAULT_BASE_URL)}`):
  your message, recent turns of this session, and the handful of memories auto-recalled for it.
- **No analytics / telemetry:** Streamlit usage stats are disabled in `.streamlit/config.toml`; BestiePrivate has no logging service.
- **Bestie can add but never delete** memories. Deletion is yours only (here, Memory tab, or `/forget <id>`).
- **Incognito toggle** keeps a chat out of the database entirely.
- **Slash commands** (`/remember`, `/recall`, `/goals`…) run 100% locally with no model call.
"""
    )
    st.json(store.stats())
    c1, c2 = st.columns(2)
    c1.download_button(
        "⬇️ Export everything (JSON)",
        data=json.dumps(store.export_all(), indent=2, default=str),
        file_name=f"bestie-export-{datetime.now():%Y%m%d-%H%M}.json",
        mime="application/json",
        use_container_width=True,
    )
    with c2:
        confirm = st.checkbox("I understand this permanently deletes all memories, goals, summaries and chats.")
        keep_sk = st.checkbox("Keep my skills", value=True)
        if st.button("🧨 Wipe all data", type="primary", disabled=not confirm, use_container_width=True):
            store.wipe_all(keep_skills=keep_sk)
            new_session()
            st.success("All data wiped.")
            st.rerun()
