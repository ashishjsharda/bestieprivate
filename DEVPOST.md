# BestiePrivate — Devpost submission copy

**Track:** Personal AI

**Tagline:** A private AI bestie that actually remembers you, with memory that never leaves your machine.

## Inspiration
Assistants either forget everything between chats or quietly store your life on someone else's server. We wanted the opposite: an assistant that builds long-term memory of the people, goals and routines that matter to you, where you can see, edit and delete every byte.

## What it does
BestiePrivate is a personal assistant with persistent, user-controlled memory and reusable skills. Tell it about your sister's birthday, your diet, or this week's goals, and it saves them to a local SQLite file. Close the app, start a new session, and it still knows. It can track goals, draft personalized messages, summarize conversations into long-term notes, and run reusable "skills" (like a daily plan or weekly review) that you create and edit yourself. A Memory tab shows everything it knows with edit/delete controls; a Privacy tab offers Incognito mode, JSON export and one-click wipe.

## How we built it
- **NVIDIA Nemotron on Nebius Token Factory** is the agent brain, called through the OpenAI-compatible API with the official OpenAI Python SDK. Users switch between Nemotron 3.5 Lightning (fast default), Nemotron 3 Super 120B, Nemotron 3 Ultra 550B and Nemotron 3 Nano live.
- **Tool calling:** Nemotron gets nine tools (save/recall memory, summarize, goals, draft message, save/run skill) and decides when to use them; a local loop executes them against SQLite and feeds results back.
- **Auto-recall:** each turn, relevant memories (keyword match + recency) plus open goals and the latest summary are injected into the prompt.
- **Streamlit** UI with per-turn latency, token and cost display so model trade-offs are visible.
- **Privacy by design:** local SQLite, no telemetry, the model can add but never delete memories, offline slash commands, Incognito mode.

## Challenges
Making memory reliable across sessions without a vector database, keeping the agent from inventing memories, and degrading gracefully when a model or endpoint doesn't support tool calling.

## Accomplishments
A complete loop (remember → restart → recall → act) that works every time, with a UI where the user stays in control of their data.

## What we learned
Smaller, fast Nemotron models handle everyday tool calling well, and larger models are worth it only for deep-reasoning skills. Routing by task keeps the assistant responsive and cheap.

## What's next
Embedding-based recall, scheduled daily briefs via Nebius Serverless Jobs, encrypted-at-rest memory, and opt-in connectors with per-tool approval.

## Built with
python · streamlit · sqlite · openai-python · nebius-token-factory · nvidia-nemotron
