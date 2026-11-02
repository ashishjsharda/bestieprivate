"""
memory.py — Local, user-owned persistent memory for BestiePrivate.

Everything lives in ONE SQLite file on your machine (default: ./data/bestie.db).
Nothing in this module talks to the network. The user can view, edit, export,
and delete every row from the UI.

Tables
------
facts      : durable things the user wants remembered ("my sister is Priya")
summaries  : compressed conversation summaries (written by summarize_recent)
skills     : reusable, user-editable prompt workflows ("weekly-review", ...)
goals      : lightweight goal / to-do tracker
messages   : chat transcript, per session (skipped entirely in Incognito mode)
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = os.getenv("BESTIE_DB_PATH", str(Path(__file__).parent / "data" / "bestie.db"))

# Tiny stopword list so keyword recall focuses on meaningful words.
_STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "your", "with", "this", "that",
    "what", "who", "how", "when", "where", "why", "was", "were", "has", "have", "had",
    "does", "did", "can", "could", "would", "should", "about", "from", "into", "they",
    "them", "their", "there", "here", "just", "like", "know", "tell", "remember",
    "recall", "please", "any", "all", "some", "our", "its", "it's", "i'm", "me", "my",
    "mine", "is", "am", "be", "of", "to", "in", "on", "at", "a", "an", "do", "i",
}

SEED_SKILLS = [
    {
        "name": "daily-plan",
        "description": "Turn my open goals plus anything I add into a prioritized plan for today.",
        "instructions": (
            "You are planning the user's day. Use their open goals and known facts. "
            "Return: (1) top 3 priorities with a one-line why, (2) a time-blocked schedule, "
            "(3) one thing to drop or delegate. Be concise and practical.\n\nExtra context from user: {input}"
        ),
    },
    {
        "name": "weekly-review",
        "description": "Reflect on the week: wins, misses, lessons, and next week's focus.",
        "instructions": (
            "Run a short weekly review for the user using their goals and memories. "
            "Sections: Wins, What slipped, One lesson, Focus for next week (max 3 items). "
            "Keep it under 200 words.\n\nWhat the user says about this week: {input}"
        ),
    },
]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9']+", (text or "").lower())
    return [w for w in words if len(w) > 2 and w not in _STOPWORDS]


class MemoryStore:
    """Thin wrapper around a local SQLite file. Safe to construct on every Streamlit rerun."""

    def __init__(self, db_path: str | None = None, seed: bool = True):
        self.db_path = db_path or DEFAULT_DB_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        if seed:
            self._seed_skills()

    # ------------------------------------------------------------------ infra
    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    tags TEXT DEFAULT '',
                    source TEXT DEFAULT 'chat',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS skills (
                    name TEXT PRIMARY KEY,
                    description TEXT DEFAULT '',
                    instructions TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    run_count INTEGER DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS goals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    due TEXT DEFAULT '',
                    status TEXT DEFAULT 'open',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    meta TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
                """
            )

    def _seed_skills(self) -> None:
        if self.list_skills():
            return
        for s in SEED_SKILLS:
            self.upsert_skill(s["name"], s["description"], s["instructions"])

    # ------------------------------------------------------------------ facts
    def add_fact(self, content: str, tags: str = "", source: str = "chat") -> int:
        content = (content or "").strip()
        if not content:
            raise ValueError("Memory content is empty.")
        # Avoid exact duplicates (case-insensitive).
        with self._conn() as c:
            row = c.execute(
                "SELECT id FROM facts WHERE lower(content) = lower(?)", (content,)
            ).fetchone()
            if row:
                return int(row["id"])
            cur = c.execute(
                "INSERT INTO facts (content, tags, source, created_at, updated_at) VALUES (?,?,?,?,?)",
                (content, tags or "", source, _now(), _now()),
            )
            return int(cur.lastrowid)

    def update_fact(self, fact_id: int, content: str, tags: str | None = None) -> bool:
        with self._conn() as c:
            if tags is None:
                cur = c.execute(
                    "UPDATE facts SET content=?, updated_at=? WHERE id=?", (content.strip(), _now(), fact_id)
                )
            else:
                cur = c.execute(
                    "UPDATE facts SET content=?, tags=?, updated_at=? WHERE id=?",
                    (content.strip(), tags, _now(), fact_id),
                )
            return cur.rowcount > 0

    def delete_fact(self, fact_id: int) -> bool:
        with self._conn() as c:
            return c.execute("DELETE FROM facts WHERE id=?", (fact_id,)).rowcount > 0

    def list_facts(self, limit: int | None = None) -> list[dict]:
        sql = "SELECT * FROM facts ORDER BY id DESC" + (f" LIMIT {int(limit)}" if limit else "")
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql).fetchall()]

    def search_facts(self, query: str, limit: int = 5) -> list[dict]:
        """Simple, transparent keyword recall: score = overlapping keywords (+ partial matches),
        ties broken by recency. Falls back to most-recent facts when the query has no keywords."""
        facts = self.list_facts()
        q = _tokens(query)
        if not q:
            return facts[:limit]
        scored = []
        for f in facts:
            hay_tokens = set(_tokens(f["content"] + " " + (f.get("tags") or "")))
            hay_text = (f["content"] + " " + (f.get("tags") or "")).lower()
            score = 0.0
            for t in q:
                if t in hay_tokens:
                    score += 1.0
                elif t.rstrip("s") and t.rstrip("s") in hay_text:  # plural / partial
                    score += 0.5
            if score > 0:
                scored.append((score, f["id"], f))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [f for _, _, f in scored[:limit]]

    # ------------------------------------------------------------- summaries
    def add_summary(self, content: str, session_id: str | None = None) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO summaries (session_id, content, created_at) VALUES (?,?,?)",
                (session_id, content.strip(), _now()),
            )
            return int(cur.lastrowid)

    def list_summaries(self, limit: int | None = None) -> list[dict]:
        sql = "SELECT * FROM summaries ORDER BY id DESC" + (f" LIMIT {int(limit)}" if limit else "")
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql).fetchall()]

    def update_summary(self, summary_id: int, content: str) -> bool:
        with self._conn() as c:
            return c.execute("UPDATE summaries SET content=? WHERE id=?", (content.strip(), summary_id)).rowcount > 0

    def delete_summary(self, summary_id: int) -> bool:
        with self._conn() as c:
            return c.execute("DELETE FROM summaries WHERE id=?", (summary_id,)).rowcount > 0

    # ----------------------------------------------------------------- skills
    def upsert_skill(self, name: str, description: str, instructions: str) -> str:
        name = re.sub(r"[^a-z0-9\-]+", "-", (name or "").strip().lower()).strip("-")
        if not name or not (instructions or "").strip():
            raise ValueError("A skill needs a name and instructions.")
        with self._conn() as c:
            exists = c.execute("SELECT 1 FROM skills WHERE name=?", (name,)).fetchone()
            if exists:
                c.execute(
                    "UPDATE skills SET description=?, instructions=?, updated_at=? WHERE name=?",
                    (description or "", instructions.strip(), _now(), name),
                )
            else:
                c.execute(
                    "INSERT INTO skills (name, description, instructions, created_at, updated_at) VALUES (?,?,?,?,?)",
                    (name, description or "", instructions.strip(), _now(), _now()),
                )
        return name

    def get_skill(self, name: str) -> dict | None:
        key = re.sub(r"[^a-z0-9\-]+", "-", (name or "").strip().lower()).strip("-")
        with self._conn() as c:
            r = c.execute("SELECT * FROM skills WHERE name=?", (key,)).fetchone()
            return dict(r) if r else None

    def list_skills(self) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM skills ORDER BY name").fetchall()]

    def bump_skill(self, name: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE skills SET run_count = run_count + 1 WHERE name=?", (name,))

    def delete_skill(self, name: str) -> bool:
        with self._conn() as c:
            return c.execute("DELETE FROM skills WHERE name=?", (name,)).rowcount > 0

    # ------------------------------------------------------------------ goals
    def add_goal(self, text: str, due: str = "") -> int:
        if not (text or "").strip():
            raise ValueError("Goal text is empty.")
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO goals (text, due, status, created_at, updated_at) VALUES (?,?,?,?,?)",
                (text.strip(), due or "", "open", _now(), _now()),
            )
            return int(cur.lastrowid)

    def list_goals(self, status: str = "open") -> list[dict]:
        with self._conn() as c:
            if status == "all":
                rows = c.execute("SELECT * FROM goals ORDER BY status DESC, id DESC").fetchall()
            else:
                rows = c.execute("SELECT * FROM goals WHERE status=? ORDER BY id DESC", (status,)).fetchall()
            return [dict(r) for r in rows]

    def set_goal_status(self, goal_id: int, status: str) -> bool:
        with self._conn() as c:
            return c.execute(
                "UPDATE goals SET status=?, updated_at=? WHERE id=?", (status, _now(), goal_id)
            ).rowcount > 0

    def delete_goal(self, goal_id: int) -> bool:
        with self._conn() as c:
            return c.execute("DELETE FROM goals WHERE id=?", (goal_id,)).rowcount > 0

    # --------------------------------------------------------------- messages
    def add_message(self, session_id: str, role: str, content: str, meta: dict | None = None) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO messages (session_id, role, content, meta, created_at) VALUES (?,?,?,?,?)",
                (session_id, role, content, json.dumps(meta or {}), _now()),
            )
            return int(cur.lastrowid)

    def get_messages(self, session_id: str, limit: int | None = None) -> list[dict]:
        with self._conn() as c:
            if limit:
                rows = c.execute(
                    "SELECT * FROM (SELECT * FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?) ORDER BY id",
                    (session_id, int(limit)),
                ).fetchall()
            else:
                rows = c.execute("SELECT * FROM messages WHERE session_id=? ORDER BY id", (session_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["meta"] = json.loads(d.get("meta") or "{}")
            except json.JSONDecodeError:
                d["meta"] = {}
            out.append(d)
        return out

    def list_sessions(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                """SELECT session_id, COUNT(*) AS n, MIN(created_at) AS started, MAX(created_at) AS last
                   FROM messages GROUP BY session_id ORDER BY last DESC"""
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_session(self, session_id: str) -> int:
        with self._conn() as c:
            return c.execute("DELETE FROM messages WHERE session_id=?", (session_id,)).rowcount

    # ---------------------------------------------------------- privacy ops
    def stats(self) -> dict:
        with self._conn() as c:
            return {
                t: c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in ("facts", "summaries", "skills", "goals", "messages")
            }

    def export_all(self) -> dict[str, Any]:
        with self._conn() as c:
            data = {
                t: [dict(r) for r in c.execute(f"SELECT * FROM {t}").fetchall()]
                for t in ("facts", "summaries", "skills", "goals", "messages")
            }
        data["exported_at"] = _now()
        data["db_path"] = os.path.abspath(self.db_path)
        return data

    def wipe_all(self, keep_skills: bool = False) -> None:
        with self._conn() as c:
            for t in ("facts", "summaries", "goals", "messages"):
                c.execute(f"DELETE FROM {t}")
            if not keep_skills:
                c.execute("DELETE FROM skills")
        # Reclaim space (VACUUM can't run inside a transaction, so use a fresh connection) so deleted text doesn't linger in free pages.
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("VACUUM")
        finally:
            conn.close()
