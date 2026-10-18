"""Exploring toward a goal: when a request names a screen the map does not have yet, explore the site in small
rounds, following first what looks most like the goal, until a matching screen exists."""
from __future__ import annotations

from ..embed import cosine
from ..graph.queries import find_targets


def goal_priority(embedder, goal: str):
    """Score for an element: how much its name and link resemble the goal (queue priority of its click)."""
    vec = embedder.embed([goal])[0]

    def score(el) -> float:
        text = f"{el.name_sample or el.name_norm or ''} {getattr(el, 'href_template', None) or ''}".strip()
        if not text:
            return 0.0
        return max(0.0, float(cosine(vec, embedder.embed([text])[0])))

    return score


def has_target(store, key: str, embedder, goal: str, text: str | None, min_score: float) -> bool:
    """Whether the map already holds a screen or workflow that matches the goal (what execute_goal would pick)."""
    try:
        hits = find_targets(store, key, embedder.embed([goal])[0], text=text or goal)
    except Exception:                                   # an empty graph has no index to query yet
        return False
    return any(t.score >= min_score for t in hits)
