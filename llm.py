"""
llm.py — Nebius Token Factory client for NVIDIA Nemotron models.

Uses the official OpenAI Python SDK pointed at Nebius Token Factory's
OpenAI-compatible endpoint. The ONLY network traffic BestiePrivate makes is the
chat-completion request built here (your message + the few memories the agent
decided were relevant). Nothing else is sent anywhere.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

# Display name -> Token Factory model ID. Override/extend with BESTIE_EXTRA_MODELS
# ("Label=model/id,Label2=model/id2") if Nebius renames or adds models.
MODELS: dict[str, str] = {
    "Nemotron 3.5 Lightning — fast": "nvidia/Nemotron-3_5-Lightning",
    "Nemotron 3 Super 120B-A12B — balanced": "nvidia/nemotron-3-super-120b-a12b",
    "Nemotron 3 Ultra 550B-A55B — deep reasoning": "nvidia/Nemotron-3-Ultra-550b-a55b",
    "Nemotron 3 Nano 30B-A3B — cheapest": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
}
for pair in filter(None, os.getenv("BESTIE_EXTRA_MODELS", "").split(",")):
    if "=" in pair:
        label, mid = pair.split("=", 1)
        MODELS[label.strip()] = mid.strip()

# Approximate list prices in USD per 1M tokens (input, output), Oct 2026.
# Used only for the on-screen cost estimate — check the Token Factory console for real billing.
PRICES: dict[str, tuple[float, float]] = {
    "nvidia/Nemotron-3_5-Lightning": (0.06, 0.24),
    "nvidia/nemotron-3-super-120b-a12b": (0.30, 0.90),
    "nvidia/Nemotron-3-Ultra-550b-a55b": (1.00, 3.00),
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B": (0.06, 0.24),
}

DEFAULT_MODEL = os.getenv("BESTIE_DEFAULT_MODEL", "nvidia/Nemotron-3_5-Lightning")
# Model used for background "utility" calls (summaries, drafts, skills). Cheap + fast by default.
UTILITY_MODEL = os.getenv("BESTIE_UTILITY_MODEL", "") or None


class MissingAPIKeyError(RuntimeError):
    """Raised when NEBIUS_API_KEY is not set."""


def api_key_present() -> bool:
    return bool(os.getenv("NEBIUS_API_KEY", "").strip())


def get_client(api_key: str | None = None) -> OpenAI:
    """Build a client. `api_key` (e.g. a visitor's own key) wins over NEBIUS_API_KEY from the env."""
    key = (api_key or "").strip() or os.getenv("NEBIUS_API_KEY", "").strip()
    if not key:
        raise MissingAPIKeyError(
            "NEBIUS_API_KEY is not set. Copy .env.example to .env and paste your key from "
            "https://tokenfactory.nebius.com (API keys), or `export NEBIUS_API_KEY=...`."
        )
    return OpenAI(
        base_url=os.getenv("NEBIUS_BASE_URL", DEFAULT_BASE_URL),
        api_key=key,
        timeout=float(os.getenv("BESTIE_TIMEOUT", "90")),
        max_retries=2,
    )


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str | None) -> str:
    """Nemotron reasoning models may emit <think>...</think> blocks. Show users only the answer."""
    if not text:
        return ""
    text = _THINK_RE.sub("", text)
    if "</think>" in text:  # opener was truncated / omitted
        text = text.split("</think>", 1)[1]
    return text.strip()


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    p = PRICES.get(model)
    if not p:
        return None
    return (prompt_tokens * p[0] + completion_tokens * p[1]) / 1_000_000


@dataclass
class CallStats:
    """Accumulates latency / token usage across all model calls in one user turn."""
    calls: int = 0
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    models: list[str] = field(default_factory=list)

    def add(self, model: str, latency: float, usage: Any) -> None:
        self.calls += 1
        self.latency_s += latency
        if model not in self.models:
            self.models.append(model)
        if usage is not None:
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0

    def cost(self) -> float | None:
        # Approximate (upper bound): attribute all tokens to the priciest model used this turn.
        prices = [PRICES[m] for m in self.models if m in PRICES]
        if not prices or len(prices) != len(self.models):
            return None
        pin = max(p[0] for p in prices)
        pout = max(p[1] for p in prices)
        return (self.prompt_tokens * pin + self.completion_tokens * pout) / 1_000_000

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "latency_s": round(self.latency_s, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "models": self.models,
            "est_cost_usd": self.cost(),
        }


def chat(
    client: OpenAI,
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    temperature: float = 0.4,
    max_tokens: int = 1200,
    stats: CallStats | None = None,
):
    """One chat-completion call against Nebius Token Factory. Returns the raw message object."""
    kwargs: dict[str, Any] = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    t0 = time.perf_counter()
    resp = client.chat.completions.create(**kwargs)
    if stats is not None:
        stats.add(model, time.perf_counter() - t0, getattr(resp, "usage", None))
    return resp.choices[0].message


def complete(client: OpenAI, model: str, system: str, user: str, stats: CallStats | None = None,
             max_tokens: int = 900, temperature: float = 0.5) -> str:
    """Convenience: single-shot text completion (no tools). Used by skills."""
    msg = chat(
        client,
        UTILITY_MODEL or model,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
        stats=stats,
    )
    return strip_reasoning(msg.content)
