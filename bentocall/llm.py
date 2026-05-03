"""OpenRouter client.

Loads OPENROUTER_API_KEY from the environment. Every call appends to a
JSONL log under `${BENTOCALL_DATA_DIR:-~/.local/share/bentocall}/calls.jsonl`
for cost/trace accounting.
"""
from __future__ import annotations

import json
import os
import pathlib
import time
from typing import List, Optional

import requests

DATA_DIR = pathlib.Path(
    os.environ.get(
        "BENTOCALL_DATA_DIR",
        os.path.expanduser("~/.local/share/bentocall"),
    )
)
CALL_LOG = DATA_DIR / "calls.jsonl"

MODEL_ALIASES = {
    "haiku": "anthropic/claude-haiku-4.5",
    "sonnet": "anthropic/claude-sonnet-4.6",
}

PRICE_PER_MTOK = {  # USD per million tokens, (input, output)
    "anthropic/claude-haiku-4.5": (1.0, 5.0),
    "anthropic/claude-sonnet-4.6": (3.0, 15.0),
}

# Anthropic prompt-caching multipliers on input price.
CACHE_WRITE_MULT = 1.25
CACHE_READ_MULT = 0.10

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

_API_KEY: Optional[str] = None


def _load_key() -> str:
    global _API_KEY
    if _API_KEY:
        return _API_KEY
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Get one at https://openrouter.ai/keys "
            "and `export OPENROUTER_API_KEY=sk-or-v1-...`."
        )
    _API_KEY = key
    return _API_KEY


def _resolve_model(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def _log_call(record: dict) -> None:
    CALL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with CALL_LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")


def M(
    prompt: str,
    model: str = "haiku",
    system: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    tag: Optional[str] = None,
    retries: int = 3,
    cache_system: bool = False,
) -> str:
    """The single neural primitive. Returns the model's text output.

    `tag` labels the call (e.g. 'leaf', 'flat', 'probe') for log filtering.
    `cache_system` marks the system prefix with Anthropic ephemeral cache_control;
    note Anthropic's minimum cacheable prefix is ~1024–2048 tokens, below which
    the request still succeeds but no cache hit is recorded.
    """
    api_key = _load_key()
    full_model = _resolve_model(model)
    messages: List[dict] = []
    if system:
        if cache_system:
            messages.append({
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }],
            })
        else:
            messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = {
        "model": full_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/aki1770-del/bentocall",
        "X-Title": "bentocall",
    }

    last_err: Optional[Exception] = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            r = requests.post(OPENROUTER_URL, headers=headers, json=body, timeout=180)
            latency = time.time() - t0
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                continue
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            in_tok = usage.get("prompt_tokens", 0)
            out_tok = usage.get("completion_tokens", 0)
            details = usage.get("prompt_tokens_details", {}) or {}
            cache_write = (
                usage.get("cache_creation_input_tokens", 0)
                or details.get("cache_creation_tokens", 0)
                or 0
            )
            cache_read = (
                usage.get("cache_read_input_tokens", 0)
                or details.get("cached_tokens", 0)
                or 0
            )
            price_in, price_out = PRICE_PER_MTOK.get(full_model, (0.0, 0.0))
            uncached_in = max(0, in_tok - cache_write - cache_read)
            cost = (
                uncached_in * price_in
                + cache_write * price_in * CACHE_WRITE_MULT
                + cache_read * price_in * CACHE_READ_MULT
                + out_tok * price_out
            ) / 1_000_000
            _log_call({
                "ts": time.time(),
                "model": full_model,
                "tag": tag,
                "prompt_chars": len(prompt),
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "cache_write_tokens": cache_write,
                "cache_read_tokens": cache_read,
                "cost_usd": round(cost, 6),
                "latency_s": round(latency, 3),
            })
            return text
        except requests.RequestException as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"OpenRouter call failed after {retries} attempts: {last_err}")
