# 🫶 BestiePrivate

**A private, always-yours AI assistant with persistent memory and reusable skills, powered by NVIDIA Nemotron on Nebius Token Factory.**

Built for the **Nebius × NVIDIA Global AI Hackathon, Personal AI Track**.

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg) ![Models: NVIDIA Nemotron](https://img.shields.io/badge/Models-NVIDIA%20Nemotron-76B900) ![Inference: Nebius Token Factory](https://img.shields.io/badge/Inference-Nebius%20Token%20Factory-blue)

BestiePrivate remembers the people, preferences, goals and routines you tell it, and it keeps that memory in **one SQLite file on your own machine**. You can see, edit, export and delete every memory. A Nemotron agent uses tool calling to save and recall memories, track goals, draft messages and run reusable skills you teach it.

---

## ✨ Features

| | |
|---|---|
| 🧠 **Persistent local memory** | Facts, conversation summaries, goals, skills and transcripts in `./data/bestie.db`. Survives restarts. |
| 🔁 **Auto-recall** | Each turn, relevant memories (keyword match + most recent) are injected into the prompt. New session, same bestie. |
| 🛠️ **Agent tools (tool calling)** | `save_memory`, `recall_memory`, `summarize_recent`, `add_goal`, `list_goals`, `complete_goal`, `draft_message`, `save_skill`, `run_skill` |
| 📚 **Reusable skills** | Named prompt workflows (`daily-plan`, `weekly-review`, or your own). Create them in the UI or just ask: *"make a skill called gift-ideas that…"* |
| 🔀 **Model switching** | Nemotron 3.5 Lightning, Nemotron 3 Super, Nemotron 3 Ultra, Nemotron 3 Nano, switchable live from the sidebar |
| 📊 **Transparent usage** | Every reply shows model, latency, tokens and estimated cost, plus a chip for each tool that ran |
| 🔒 **Privacy controls** | Memory viewer and editor, Incognito mode, JSON export, one-click wipe, model can add but **never delete** memories, Streamlit telemetry off |
| ⚡ **Offline slash commands** | `/remember`, `/recall`, `/forget`, `/goals`, `/goal`, `/done`, `/skills`, `/run`, `/summarize`, `/help`. Memory ops run locally with no model call. |

## 🏗️ Architecture

```
┌──────────────────────── your machine ─────────────────────────┐
│  Streamlit UI (app.py)                                         │
│    Chat · Memory viewer/editor · Skills & Goals · Privacy      │
│            │                                                   │
│  Agent loop (agent.py)                                         │
│    slash command? ──► run locally (no model call)              │
│    else: system prompt + auto-recalled memories ──┐            │
│            ▲            tool calls ◄──────────────┤            │
│            │                                      │            │
│  Tools (tools.py) ──► MemoryStore (memory.py) ──► data/bestie.db (SQLite)
└───────────────────────────────────────────────────┼────────────┘
                                                    │ HTTPS (OpenAI-compatible)
                                     ┌──────────────▼──────────────┐
                                     │  Nebius Token Factory        │
                                     │  NVIDIA Nemotron models      │
                                     └──────────────────────────────┘
```

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI: chat, memory viewer/editor, skills & goals, privacy tab |
| `agent.py` | Agent loop: prompt building, auto-recall, tool-calling loop, slash commands, graceful fallback |
| `tools.py` | Tool schemas (what Nemotron sees) and local implementations |
| `memory.py` | SQLite memory store: facts, summaries, skills, goals, messages; export and wipe |
| `llm.py` | Nebius Token Factory client (OpenAI SDK), model catalog, cost estimate, `<think>` stripping |
| `smoke_test.py` | Verifies your key, lists Nemotron models, tests chat and tool calling |
| `tests/` | Offline tests with a fake model client (no API key needed) |

---

## 🚀 Setup

### 1. Get a Nebius Token Factory API key
1. Sign in at **https://tokenfactory.nebius.com** (hackathon credits apply to your account).
2. Open **API keys** → **Create API key** → copy it.
3. In the **Models** catalog, confirm the Nemotron models you want are enabled for your account.

### 2. Install
Requires Python 3.10+.

```bash
git clone <your-repo-url> bestieprivate && cd bestieprivate
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # then paste your key into NEBIUS_API_KEY
```

### 3. Verify (30 seconds)
```bash
python smoke_test.py
```
This lists the Nemotron models your key can reach, runs a chat, and checks that tool calling works.

### 4. Run
```bash
streamlit run app.py
```
Open http://localhost:8501.

### Tests (no API key needed)
```bash
python -m unittest discover -s tests -v
```

---

## 🟩 How NVIDIA Nemotron and Nebius Token Factory are used

- **Inference:** every model call goes through the official `openai` Python SDK pointed at Nebius Token Factory's OpenAI-compatible endpoint (`base_url="https://api.tokenfactory.nebius.com/v1/"`, key from `NEBIUS_API_KEY`). See `llm.py`.
- **Nemotron as the agent brain:** the selected Nemotron model receives the system prompt, auto-recalled memories and nine tool schemas, then decides when to call `save_memory`, `recall_memory`, `add_goal`, `run_skill`, etc. (`agent.py`). Tool results are fed back to the model until it answers (up to 5 rounds).
- **Nemotron for skills:** `summarize_recent`, `draft_message` and `run_skill` make their own focused Nemotron calls. Set `BESTIE_UTILITY_MODEL` to route these background calls to a cheaper model while the main chat uses a bigger one.
- **Right model per job:**

| Model (Token Factory ID) | Used for |
|---|---|
| `nvidia/Nemotron-3_5-Lightning` | **Default.** Fast, cheap everyday chat and tool calling |
| `nvidia/nemotron-3-super-120b-a12b` | Balanced: better multi-step tool use and planning |
| `nvidia/Nemotron-3-Ultra-550b-a55b` | Deep reasoning (weekly reviews, hard planning) |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` | Cheapest option for utility calls |

- **Reasoning output:** Nemotron reasoning traces (`<think>…</think>`) are stripped before display.
- **Resilience:** if a model or endpoint rejects tool calling, the agent falls back to plain chat and notes it in the UI instead of crashing.
- **Token Factory made this fast to build:** one OpenAI-compatible endpoint meant zero custom SDK work, and switching between four Nemotron sizes is a dropdown, not a redeploy.

> Model IDs and prices may change. Run `python smoke_test.py` to see what's live for your key, and add models without code changes via `BESTIE_EXTRA_MODELS="Label=model/id"`.

## 🔒 Privacy model

- Memory is a local SQLite file (`data/bestie.db`, git-ignored). No cloud database, no analytics, no logging service.
- The only network call is the chat completion to Nebius Token Factory: your message, the last few turns of the current session, and the memories auto-recalled for that message.
- Streamlit usage statistics are disabled (`.streamlit/config.toml`).
- The model can **add** memories but has no delete tool. Deletion is user-only.
- **Incognito** mode keeps a conversation out of the database.
- **Export** everything as JSON or **wipe** everything from the Privacy tab.

## 🎬 Demo path (works every time)

1. `streamlit run app.py`
2. Say: *"Remember my sister Priya's birthday is March 3 and she loves orchids."* → 🔧 `save_memory` chip appears; Memory tab shows the fact.
3. Say: *"Add a goal: buy Priya's gift by March 1."* → 🔧 `add_goal`
4. Click **🔄 New session** (or stop and restart Streamlit).
5. Ask: *"What do you know about my sister?"* → Bestie recalls Priya, March 3, orchids.
6. Say: *"Draft a birthday text to Priya."* → 🔧 `draft_message`, personalized from memory.
7. Memory tab → edit or delete the fact. Privacy tab → export JSON.

---

## 📝 Feedback notes (template — fill in after testing)

**Nebius Token Factory**
- Onboarding (sign-up → first successful call): ___ minutes. Friction points: ___
- Docs quality / OpenAI compatibility (tools, `tool_choice`, streaming, usage fields): ___
- Model catalog discoverability (finding exact model IDs): ___
- Console: key management, usage/cost visibility: ___
- Errors encountered (status code, message, how resolved): ___

**NVIDIA Nemotron models**
| Model | Avg latency (simple turn) | Avg latency (tool turn) | Tool-call reliability | Quality notes |
|---|---|---|---|---|
| Nemotron 3.5 Lightning | ___ s | ___ s | __/10 calls correct | ___ |
| Nemotron 3 Super 120B | ___ s | ___ s | __/10 | ___ |
| Nemotron 3 Ultra 550B | ___ s | ___ s | __/10 | ___ |

- Reasoning traces (`<think>`): helpful / noisy? ___
- Instruction following on "save one fact per call": ___

**Cost**
- Total spend during development: $___ over ___ requests
- Typical cost per chat turn (from in-app estimate): $___

**What would make it better**
- ___

## 🎥 Demo video script (≤ 3:00)

| Time | Show | Say |
|---|---|---|
| 0:00–0:15 | Title + app home | "Every AI assistant forgets you or keeps your life on someone else's server. BestiePrivate remembers you, and the memory stays on your machine." |
| 0:15–0:35 | Sidebar model picker, `.env`, `llm.py` base_url | "It runs NVIDIA Nemotron on Nebius Token Factory through the OpenAI-compatible API. I can switch Lightning, Super, or Ultra live." |
| 0:35–1:05 | Chat: tell it about Priya + add goal | "I just talk. Nemotron decides to call `save_memory` and `add_goal`. Each chip is a tool that ran locally. Latency, tokens and cost are shown on every turn." |
| 1:05–1:35 | Memory tab → edit/delete; then **New session** | "Here's everything it knows, editable and deletable. The model can add memories but never delete them. Now a brand-new session…" |
| 1:35–2:00 | Ask "What do you know about my sister?" | "…and it remembers. Memory persisted across sessions in local SQLite." |
| 2:00–2:30 | "Draft a birthday text to Priya" + `/run daily-plan` | "Skills are reusable workflows. This draft is personalized from memory. Skills are user-editable." |
| 2:30–2:50 | Switch to Ultra for weekly-review; Privacy tab (Incognito, export, wipe) | "Big model for deep thinking, small model for everyday speed. Privacy tab: incognito, export, wipe." |
| 2:50–3:00 | README / GitHub | "Open source, MIT licensed. Your bestie, your data." |

---

## 🗺️ Roadmap
- Embedding-based recall (Nebius-hosted embedding model) alongside keyword recall
- Scheduled daily brief via Nebius Serverless Jobs
- Encrypted-at-rest memory file (SQLCipher)
- Connectors (calendar, email) gated by per-tool user approval

## License
[MIT](LICENSE)
