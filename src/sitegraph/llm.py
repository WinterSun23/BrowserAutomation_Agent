from __future__ import annotations

import base64
import json
import re
from typing import Callable, Protocol

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


class LLM(Protocol):
    def complete(self, system: str, user: str, image_png: bytes | None = None) -> str: ...


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
    except Exception:
        if default is not None:
            return default
        raise


class LiteLLMClient:
    """Any OpenAI-compatible model via LiteLLM, e.g. model='openai/gpt-4o-mini' + api_base."""

    def __init__(self, model: str, api_base: str | None = None, api_key: str | None = None, temperature: float = 0.0):
        self.model, self.api_base, self.api_key, self.temperature = model, api_base, api_key, temperature

    def complete(self, system: str, user: str, image_png: bytes | None = None) -> str:
        import litellm

        content: object = user
        if image_png is not None:
            b64 = base64.b64encode(image_png).decode()
            content = [{"type": "text", "text": user},
                       {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]
        resp = litellm.completion(model=self.model, api_base=self.api_base, api_key=self.api_key,
                                  temperature=self.temperature,
                                  messages=[{"role": "system", "content": system},
                                            {"role": "user", "content": content}])
        return resp.choices[0].message.content or ""


class FakeLLM:
    def __init__(self, responder: Callable[[str, str], str] | None = None):
        self.responder = responder
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, image_png: bytes | None = None) -> str:
        self.calls.append((system, user))
        return self.responder(system, user) if self.responder else "{}"
