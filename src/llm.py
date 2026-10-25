"""Thin LLM adapter: one canonical message format, two back ends.

Canonical messages (what the agent stores):
    {"role": "user", "content": str}
    {"role": "assistant", "text": str, "tool_calls": [{"id", "name", "args"}]}
    {"role": "tool", "id": str, "name": str, "content": str}

Providers (LLM_PROVIDER):
    anthropic  - Claude via the `anthropic` SDK (ANTHROPIC_API_KEY, LLM_MODEL)
    openai     - any OpenAI-compatible endpoint via the `openai` SDK (OPENAI_API_KEY,
                 OPENAI_BASE_URL, LLM_MODEL): OpenAI, Gemini, Groq, OpenRouter, Ollama ...
"""
import json
import os
import re
import time
from dataclasses import dataclass, field

from . import config  # noqa: F401  (loads .env before any client is created)


@dataclass
class LLMReply:
    text: str = ""
    tool_calls: list = field(default_factory=list)


class LLMError(RuntimeError):
    pass


def _retry_delay(exc, attempt: int) -> float:
    """Honour 'try again in 3.2s' / '1m5s' hints from the provider, else exponential back-off."""
    m = re.search(r"try again in (?:(\d+)m)?(?:(\d+(?:\.\d+)?)(ms|s))?", str(exc))
    if m and (m.group(1) or m.group(2)):
        secs = int(m.group(1) or 0) * 60 + float(m.group(2) or 0) / (1000 if m.group(3) == "ms" else 1)
        return min(secs + 1, 90)
    return min(2 ** (attempt + 1), 60)


def _is_daily_cap(exc) -> bool:
    t = str(exc).lower()
    return "per day" in t or "(tpd)" in t or "requests per day" in t or "daily" in t


class LLM:
    def __init__(self, provider: str = None, model: str = None):
        self.provider = (provider or os.environ.get("LLM_PROVIDER")
                         or ("anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openai")).lower()
        default = "claude-sonnet-5-5" if self.provider == "anthropic" else "gpt-4o-mini"
        self.model = model or os.environ.get("LLM_MODEL") or default
        # comma-separated models to rotate to when the current one hits its DAILY token cap
        self.fallbacks = [m.strip() for m in os.environ.get("LLM_FALLBACK_MODELS", "").split(",") if m.strip()]
        self._client = None

    # ---- client ---------------------------------------------------------------------
    def _get_client(self):
        if self._client is None:
            try:
                if self.provider == "anthropic":
                    import anthropic
                    self._client = anthropic.Anthropic()
                else:
                    import openai
                    self._client = openai.OpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None)
            except Exception as exc:
                raise LLMError(f"Could not start the '{self.provider}' LLM client: {exc}. "
                               "Check .env (see .env.example) and `pip install -r requirements.txt`.") from exc
        return self._client

    # ---- public ---------------------------------------------------------------------
    def complete(self, system: str, messages: list, tools: list = None, max_tokens: int = 1800) -> LLMReply:
        last = None
        for attempt in range(7):  # rate limits and flaky networks: back off, then give up cleanly
            try:
                if self.provider == "anthropic":
                    return self._anthropic(system, messages, tools, max_tokens)
                return self._openai(system, messages, tools, max_tokens)
            except LLMError:
                raise
            except Exception as exc:
                last = exc
                if _is_daily_cap(exc) and self.fallbacks:  # waiting would take hours: switch model now
                    self.model = self.fallbacks.pop(0)
                    print(f"[llm] daily cap reached, switching to {self.model}", flush=True)
                    continue
                transient = any(s in f"{type(exc).__name__} {exc}".lower()
                                for s in ("rate", "429", "overload", "timeout", "connection", "529", "503"))
                if not transient:
                    break
                time.sleep(_retry_delay(exc, attempt))
        raise LLMError(f"LLM call failed: {last}") from last

    # ---- Anthropic ------------------------------------------------------------------
    def _anthropic(self, system, messages, tools, max_tokens) -> LLMReply:
        msgs = []
        for m in messages:
            if m["role"] == "user":
                msgs.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                blocks = []
                if m.get("text"):
                    blocks.append({"type": "text", "text": m["text"]})
                for tc in m.get("tool_calls", []):
                    blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["args"]})
                msgs.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "..."}]})
            else:  # tool result -> user turn, merged with neighbours
                block = {"type": "tool_result", "tool_use_id": m["id"], "content": m["content"]}
                if msgs and msgs[-1]["role"] == "user" and isinstance(msgs[-1]["content"], list):
                    msgs[-1]["content"].append(block)
                else:
                    msgs.append({"role": "user", "content": [block]})
        kwargs = dict(model=self.model, system=system, messages=msgs, max_tokens=max_tokens, temperature=0)
        if tools:
            kwargs["tools"] = [{"name": t["name"], "description": t["description"],
                                "input_schema": t["parameters"]} for t in tools]
        resp = self._get_client().messages.create(**kwargs)
        out = LLMReply()
        for b in resp.content:
            if b.type == "text":
                out.text += b.text
            elif b.type == "tool_use":
                out.tool_calls.append({"id": b.id, "name": b.name, "args": dict(b.input)})
        return out

    # ---- OpenAI-compatible ----------------------------------------------------------
    def _openai(self, system, messages, tools, max_tokens) -> LLMReply:
        msgs = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "user":
                msgs.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                msg = {"role": "assistant", "content": m.get("text") or None}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [{"id": tc["id"], "type": "function",
                                          "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}}
                                         for tc in m["tool_calls"]]
                msgs.append(msg)
            else:
                msgs.append({"role": "tool", "tool_call_id": m["id"], "content": m["content"]})
        kwargs = dict(model=self.model, messages=msgs, max_tokens=max_tokens, temperature=0)
        if "gpt-oss" in self.model:
            kwargs["reasoning_effort"] = os.environ.get("LLM_REASONING_EFFORT", "low")
        if tools:
            kwargs["tools"] = [{"type": "function", "function": {
                "name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
                for t in tools]
        resp = self._get_client().chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        out = LLMReply(text=msg.content or "")
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            out.tool_calls.append({"id": tc.id, "name": tc.function.name, "args": args})
        return out
