"""Exploring toward a goal: when a request names a screen the map does not have yet, explore the site in small
rounds, following first what looks most like the goal, until a matching screen exists."""
from __future__ import annotations

import re

from ..embed import cosine
from ..graph.queries import Target, find_targets
from .frontier import PAGINATION

PAGINATION_BONUS = 0.2
_WORD = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset("""a an the of to in on at for from by with and or is are be me my i you your it its this that
these those what which who how much many please can could would will show find take go open get give tell see view
display bring navigate visit page screen site website want need let us some any all list""".split())


def goal_words(goal: str) -> set[str]:
    """The goal's content words: lower case, stopwords and request filler ('find', 'show me', 'take me to') removed."""
    return {w for w in _WORD.findall((goal or "").lower()) if w not in STOPWORDS and len(w) > 1}


def _has_word(hay: set[str], w: str) -> bool:
    return w in hay or w.rstrip("s") in hay or w + "s" in hay


def match_share(goal: str, title: str, url: str, text: str) -> float:
    """Share of the goal's words found in the page's title, URL and visible text (0 when the goal has none)."""
    words = goal_words(goal)
    if not words:
        return 0.0
    hay = set(_WORD.findall(f"{title or ''} {url or ''} {text or ''}".lower()))
    return sum(_has_word(hay, w) for w in words) / len(words)


def quick_match(goal: str, title: str, url: str, text: str, min_share: float) -> bool:
    return bool(goal_words(goal)) and match_share(goal, title, url, text) >= min_share


def state_text(store, key: str, state_gid: str) -> tuple[str, str, str]:
    """(title, url, visible text) of a state from the graph: its page's title and example URL and the names of its
    elements and list items (the graph keeps no page text). Empty strings when it cannot be read."""
    try:
        rows = store.ro_query(key, """
            MATCH (p:Page)-[:HAS_STATE]->(s:State {gid:$g})
            OPTIONAL MATCH (s)-[:COMPOSED_OF]->(:ComponentState)-[:CONTAINS]->(x:Element)
            RETURN p.title, p.example_urls[0], coalesce(s.summary, ''), collect(x.name_sample),
                   collect(x.sample_items)""", {"g": state_gid})
    except Exception:
        return "", "", ""
    if not rows:
        return "", "", ""
    title, url, summary, names, items = rows[0]
    flat = [i for lst in items or [] for i in (lst or [])]
    return title or "", url or "", " ".join([summary] + [n for n in names if n] + flat)


def found_target(store, key: str, embedder, goal: str, text: str | None, settings,
                 page_texts: dict[str, str] | None = None) -> Target | None:
    """The map's target for the goal, held to a stricter test than `execute_goal`'s: it counts only with a score of
    at least `goal_found_score`, or at least `min_target_score` plus the goal's words on its page
    (`goal_words_min`). `page_texts` (state gid -> visible text) is used when known; else the graph's names."""
    try:
        hits = find_targets(store, key, embedder.embed([goal])[0], text=text or goal)
    except Exception:                                   # an empty graph has no index to query yet
        return None
    for t in hits:
        if t.score >= settings.goal_found_score:
            return t
        if t.score < settings.min_target_score:
            continue
        title, url, gtext = state_text(store, key, t.gid) if t.kind == "State" else ("", "", "")
        page = (page_texts or {}).get(t.gid) or gtext or t.text
        if quick_match(goal, title, url, f"{t.text} {page}", settings.goal_words_min):
            return t
    return None


def goal_priority(embedder, goal: str, unseen_item: bool = False):
    """Score for an element: how much its name and link resemble the goal (queue priority of its click). With
    `unseen_item` (the goal names a list item the map has not seen), pagination links get a bonus: the item is
    probably on a later page of a list."""
    vec = embedder.embed([goal])[0]

    def score(el) -> float:
        name = el.name_sample or el.name_norm or ""
        text = f"{name} {getattr(el, 'href_template', None) or ''}".strip()
        if not text:
            return 0.0
        base = max(0.0, float(cosine(vec, embedder.embed([text])[0])))
        if unseen_item and PAGINATION.search(name):
            base += PAGINATION_BONUS
        return base

    return score


def has_target(store, key: str, embedder, goal: str, text: str | None, min_score: float) -> bool:
    """Whether the map already holds a screen or workflow that matches the goal (what execute_goal would pick)."""
    try:
        hits = find_targets(store, key, embedder.embed([goal])[0], text=text or goal)
    except Exception:                                   # an empty graph has no index to query yet
        return False
    return any(t.score >= min_score for t in hits)
