from __future__ import annotations

import json
from dataclasses import dataclass

from ..llm import complete_json

FORMATS = ["video", "guide", "clickthrough", "gif", "screenshot"]
SCOPES = ["workflow", "goal", "state", "transition", "feature"]
_SYSTEM = "You turn a user's request into a capture plan for a website demo agent."


@dataclass
class DemoSpec:
    formats: list[str]
    scope_kind: str
    target: str
    audience: str = "general"
    max_seconds: int = 60


_FORMAT_WORDS = [("gif", "gif"), ("screenshot", "screenshot"), ("screen shot", "screenshot"),
                 ("click-through", "clickthrough"), ("clickthrough", "clickthrough"), ("click through", "clickthrough"),
                 ("guide", "guide"), ("step-by-step", "guide"), ("step by step", "guide"), ("video", "video")]


def _formats_named_in(text: str) -> list[str]:
    """Fallback when no LLM is available: formats the user literally asked for."""
    low = text.lower()
    found: list[str] = []
    for word, fmt in _FORMAT_WORDS:
        if word in low and fmt not in found:
            found.append(fmt)
    return found


def interpret(llm, text: str, prefs: dict | None = None) -> DemoSpec:
    prefs = prefs or {}
    default = {"formats": _formats_named_in(text) or [prefs.get("demo.format", "video")],
               "scope_kind": "goal", "target": text,
               "audience": "general", "max_seconds": 60}
    data = complete_json(llm, _SYSTEM,
                         f"Request: {text}\nUser preferences: {json.dumps(prefs)}\n"
                         f"Return JSON with keys: formats (subset of {FORMATS}), scope_kind (one of {SCOPES}; "
                         "workflow=a saved automation, goal=how to reach something, state=one screen, "
                         "transition=a small interaction like opening a menu, feature=an area of the site), "
                         "target (short description of what to show), audience, max_seconds.", default=default)
    if not isinstance(data, dict):
        data = default
    formats = [f for f in data.get("formats") or [] if f in FORMATS] or default["formats"]
    scope = data.get("scope_kind") if data.get("scope_kind") in SCOPES else "goal"
    try:
        seconds = int(data.get("max_seconds") or 60)
    except (TypeError, ValueError):
        seconds = 60
    return DemoSpec(formats=formats, scope_kind=scope, target=str(data.get("target") or text),
                    audience=str(data.get("audience") or "general"), max_seconds=seconds)
