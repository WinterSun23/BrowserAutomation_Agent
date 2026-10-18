from __future__ import annotations

import json
import sys


class Events:
    def __init__(self, redis_client, prefix: str = "events"):
        self.r, self.prefix = redis_client, prefix

    def channel(self, uid: str) -> str:
        return f"{self.prefix}:{uid}"

    def publish(self, uid: str, event: dict) -> None:
        self.r.publish(self.channel(uid), json.dumps(event))


class NullEvents:
    def channel(self, uid: str) -> str:
        return f"events:{uid}"

    def publish(self, uid: str, event: dict) -> None:
        pass


def _stderr(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


class ConsoleEvents:
    """Progress lines on stderr for `--verbose` (stdout stays the JSON result). Unknown event types are ignored."""

    def __init__(self, write=_stderr):
        self.write = write

    def channel(self, uid: str) -> str:
        return f"events:{uid}"

    def publish(self, uid: str, event: dict) -> None:
        line = self.format(event)
        if line:
            self.write(line)

    @staticmethod
    def format(e: dict) -> str | None:
        t = e.get("type")
        if t == "state":
            return (f"+ state {e.get('title') or '?'} ({e.get('url_template') or ''}) "
                    f"[{'new' if e.get('new') else 'known'}] by {e.get('agent') or '?'}")
        if t == "step":
            return f"  {'ok' if e.get('ok', True) else 'FAILED'} {e.get('label', '')}" + (
                f" ({e['error']})" if e.get("error") else "")
        if t == "claim":
            return f"  {e.get('agent', '?')} takes {e.get('kind', '?')} {e.get('what', '')}".rstrip()
        if t == "spawn":
            return f"* started explorer {e.get('agent', '?')} (backlog {e.get('backlog', '?')})"
        if t == "explore_done":
            return (f"explore done: {e.get('states', 0)} states, {e.get('actions', 0)} actions, "
                    f"{e.get('transitions', 0)} transitions, {e.get('blocked', 0)} blocked")
        if t == "task":
            return f"task {e.get('status', '?')}" + (f": {e['text']}" if e.get("text") else "")
        return None
