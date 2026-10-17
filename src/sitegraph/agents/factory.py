"""The one place a Navigator is built. Everything that needs a Navigator (the Orchestrator, the chat agent's tools, the
demo recorder) goes through `make_navigator`, so git recording, secrets and other per-agent wiring added later apply
to every agent the same way."""
from __future__ import annotations

from .navigator import Navigator


def make_navigator(store, key, domain, session, settings, embedder, llm=None, *, blobs=None, events=None,
                   agent_id: str = "navigator-1", task_id: str | None = None, ops=None, writer=None, secrets=None,
                   notes=None, escalate=None) -> Navigator:
    """`writer`: the MapWriter (a RecordingMapWriter when the map is under version control); `secrets`: a callable
    `var -> value | None` backed by the vault; `escalate`: the recovery agent."""
    return Navigator(store, key, domain, session, settings, embedder, llm, blobs, events, agent_id=agent_id,
                     task_id=task_id, ops=ops, escalate=escalate, secrets=secrets, writer=writer,
                     notes=notes)
