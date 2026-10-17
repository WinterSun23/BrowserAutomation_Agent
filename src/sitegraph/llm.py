from __future__ import annotations

import base64
import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

logger = logging.getLogger(__name__)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
NUM_RETRIES = 3  # LiteLLM retries rate limits and transient errors with backoff


class LLM(Protocol):
    def complete(self, system: str, user: str, image_png: bytes | None = None) -> str: ...


# ---------- tool calling (used by the agents) ----------
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    arguments_error: str | None = None  # set when the model sent arguments that are not a JSON object


@dataclass
class ChatMessage:
    role: str  # system | user | assistant | tool
    content: str | None = None
    tool_calls: list[ToolCall] | None = None  # on assistant messages
    tool_call_id: str | None = None  # on tool messages
    name: str | None = None  # on tool messages: which tool answered
    usage: dict | None = None  # on assistant messages from a real model: token counts


class ToolLLM(Protocol):
    def chat(self, messages: list[ChatMessage], tools: list[dict], image_png: bytes | None = None) -> ChatMessage: ...


def to_litellm(msg: ChatMessage) -> dict:
    """ChatMessage -> the OpenAI-style message dict LiteLLM expects."""
    if msg.role == "tool":
        return {"role": "tool", "content": msg.content or "", "tool_call_id": msg.tool_call_id, "name": msg.name}
    d: dict = {"role": msg.role, "content": msg.content}
    if msg.tool_calls:
        d["tool_calls"] = [{"id": c.id, "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                           for c in msg.tool_calls]
    return d


def from_litellm(message, usage=None) -> ChatMessage:
    """A LiteLLM / OpenAI response message -> ChatMessage; bad tool arguments never raise."""
    calls: list[ToolCall] = []
    for c in getattr(message, "tool_calls", None) or []:
        raw, args, err = c.function.arguments, {}, None
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    args = parsed
                else:
                    err = "arguments must be a JSON object"
            except (json.JSONDecodeError, TypeError):
                err = "arguments are not valid JSON"
        calls.append(ToolCall(id=c.id, name=c.function.name, arguments=args, arguments_error=err))
    tokens = getattr(usage, "total_tokens", None)
    return ChatMessage("assistant", content=getattr(message, "content", None) or None, tool_calls=calls or None,
                       usage={"total_tokens": tokens} if isinstance(tokens, int) else None)


def parse_json(text: str | None):
    if not text:
        raise ValueError("empty LLM output")
    t = text.strip()
    m = _FENCE.search(t)
    if m:
        t = m.group(1).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    for open_, close in (("{", "}"), ("[", "]")):
        i, j = t.find(open_), t.rfind(close)
        if i != -1 and j > i:
            try:
                return json.loads(t[i:j + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON in LLM output: {text[:200]!r}")


def complete_json(llm: LLM, system: str, user: str, default=None, image_png: bytes | None = None):
    try:
        return parse_json(llm.complete(system + "\nRespond with a single JSON value only, no prose.", user, image_png))
    except Exception as exc:
        if default is not None:
            if has_model(llm):  # offline defaults are expected; a real model failing is not
                logger.warning("LLM call failed, using the built-in default (%s: %s)", type(exc).__name__,
                               str(exc)[:200])
            return default
        raise


class LiteLLMClient:
    """Any OpenAI-compatible model via LiteLLM, e.g. model='openai/gpt-4o-mini' + api_base."""

    def __init__(self, model: str, api_base: str | None = None, api_key: str | None = None, temperature: float = 0.0):
        self.model, self.api_base, self.api_key, self.temperature = model, api_base, api_key, temperature
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def _count(self, resp) -> None:
        u = getattr(resp, "usage", None)
        self.usage["calls"] += 1
        if u is not None:
            self.usage["prompt_tokens"] += int(getattr(u, "prompt_tokens", 0) or 0)
            self.usage["completion_tokens"] += int(getattr(u, "completion_tokens", 0) or 0)

    def complete(self, system: str, user: str, image_png: bytes | None = None) -> str:
        import litellm

        content: object = user
        if image_png is not None:
            b64 = base64.b64encode(image_png).decode()
            content = [{"type": "text", "text": user},
                       {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]
        resp = litellm.completion(model=self.model, api_base=self.api_base, api_key=self.api_key,
                                  temperature=self.temperature, num_retries=NUM_RETRIES,
                                  messages=[{"role": "system", "content": system},
                                            {"role": "user", "content": content}])
        self._count(resp)
        return resp.choices[0].message.content or ""

    def chat(self, messages: list[ChatMessage], tools: list[dict], image_png: bytes | None = None) -> ChatMessage:
        import litellm

        payload = [to_litellm(m) for m in messages]
        if image_png is not None and payload and payload[-1]["role"] == "user":
            b64 = base64.b64encode(image_png).decode()
            payload[-1]["content"] = [{"type": "text", "text": payload[-1]["content"] or ""},
                                      {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]
        resp = litellm.completion(model=self.model, api_base=self.api_base, api_key=self.api_key,
                                  temperature=self.temperature, num_retries=NUM_RETRIES, messages=payload,
                                  tools=tools or None)
        self._count(resp)
        return from_litellm(resp.choices[0].message, getattr(resp, "usage", None))


class OfflineLLM:
    """Used when no model is configured: every call fails, so `complete_json` returns its built-in defaults
    (keyword routing, title-based summaries, plain captions). The agent still works, with less polish."""

    def complete(self, system: str, user: str, image_png: bytes | None = None) -> str:
        raise RuntimeError("no LLM configured (set LLM_API_KEY, or LLM_API_BASE for a local server)")

    def chat(self, messages: list[ChatMessage], tools: list[dict], image_png: bytes | None = None) -> ChatMessage:
        raise RuntimeError("no LLM configured (set LLM_API_KEY, or LLM_API_BASE for a local server)")


class ScriptedLLM:
    """Test double for tool-calling agents: replays prepared replies in order. Each item is a ChatMessage or a
    callable (messages, tools) -> ChatMessage. `calls` records what the agent sent."""

    def __init__(self, script: list):
        self.script = list(script)
        self.calls: list[tuple[list[ChatMessage], list[dict]]] = []
        self.images: list[bytes | None] = []

    def chat(self, messages: list[ChatMessage], tools: list[dict], image_png: bytes | None = None) -> ChatMessage:
        self.calls.append((list(messages), tools))
        self.images.append(image_png)
        if not self.script:
            raise RuntimeError("ScriptedLLM script exhausted")
        step = self.script.pop(0)
        return step(messages, tools) if callable(step) else step


class FakeLLM:
    def __init__(self, responder: Callable[[str, str], str] | None = None):
        self.responder = responder
        self.calls: list[tuple[str, str]] = []
        self.images: list[bytes | None] = []

    def complete(self, system: str, user: str, image_png: bytes | None = None) -> str:
        self.calls.append((system, user))
        self.images.append(image_png)
        return self.responder(system, user) if self.responder else "{}"


# ---------- wrappers must not hide what is underneath ----------
def unwrap_llm(llm):
    """The real client under any number of wrappers (TracingLLM and friends expose it as `.inner`)."""
    while hasattr(llm, "inner"):
        llm = llm.inner
    return llm


def has_model(llm) -> bool:
    """A real model is configured (not missing, not the offline fallback), also through wrappers."""
    return llm is not None and not isinstance(unwrap_llm(llm), OfflineLLM)


def can_call_tools(llm) -> bool:
    """The agents need a real model that supports tool calling (a wrapper only has `chat` when its inner model does)."""
    return has_model(llm) and hasattr(llm, "chat")


class TracingLLM:
    """Wraps an LLM and appends every call to a per-run JSONL trace (thread-local current trace). Tool arguments that
    `redact(tool_name, args)` hides (typed values, secrets) are replaced in the trace; image bytes are never written."""

    def __init__(self, inner, trace_dir: Path, redact: Callable[[str, dict], dict] | None = None):
        self.inner, self.dir, self.redact = inner, Path(trace_dir), redact
        self.dir.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.calls = 0

    @property
    def usage(self) -> dict:
        return getattr(self.inner, "usage", {"calls": self.calls})

    def start_trace(self, run_gid: str) -> Path:
        path = self.dir / f"{run_gid.replace(':', '_')}.jsonl"
        self._local.path = path
        return path

    def stop_trace(self) -> Path | None:
        path = getattr(self._local, "path", None)
        self._local.path = None
        return path

    def _write(self, record: dict) -> None:
        path = getattr(self._local, "path", None)
        if path is not None:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")

    def complete(self, system: str, user: str, image_png: bytes | None = None) -> str:
        t0 = time.monotonic()
        out = self.inner.complete(system, user, image_png)
        self.calls += 1
        self._write({"ts": time.time(), "ms": int((time.monotonic() - t0) * 1000), "kind": "complete",
                     "system": system[:4000], "user": user[:4000], "image": image_png is not None,
                     "output": out[:4000]})
        return out

    def _calls(self, calls) -> list[dict]:
        return [{"name": c.name, "arguments": self.redact(c.name, c.arguments) if self.redact else c.arguments}
                for c in calls or []]

    def _summary(self, m: ChatMessage) -> dict:
        d = {"role": m.role, "content": (m.content or "")[:2000]}
        if m.tool_calls:
            d["tool_calls"] = self._calls(m.tool_calls)
        if m.role == "tool":
            d["tool"] = m.name
        return d

    @property
    def chat(self):
        """Present only when the wrapped model can call tools (so `hasattr(llm, 'chat')` stays truthful)."""
        inner_chat = getattr(self.inner, "chat")        # AttributeError here makes hasattr(self, 'chat') False
        def traced(messages: list[ChatMessage], tools: list[dict], image_png: bytes | None = None) -> ChatMessage:
            t0 = time.monotonic()
            out = inner_chat(messages, tools, image_png)
            self.calls += 1
            self._write({"ts": time.time(), "ms": int((time.monotonic() - t0) * 1000), "kind": "chat",
                         "messages": [self._summary(m) for m in messages[-6:]], "image": image_png is not None,
                         "output": self._summary(out), "usage": out.usage})
            return out
        return traced
