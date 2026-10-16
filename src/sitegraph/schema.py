from __future__ import annotations

from .db import GraphStore

_DEVICE_EXACT = ["State", "Page", "Component", "ComponentState", "Element", "Workflow", "WStep",
                 "Intent", "Feature", "Run", "Step", "Observation", "DemoRequest", "Storyboard", "Shot", "Render"]
_DEVICE_REL = [("ACTION", "gid"), ("TRANSITION", "gid"), ("ACTION", "el_gid")]
_DEVICE_FULLTEXT = [("Element", "name_sample"), ("Page", "title")]
_DEVICE_VECTOR = ["State", "Component", "Element", "Intent", "Workflow", "DemoRequest", "Feature"]
_DEVICE_UNIQUE = ["State", "Page", "Component", "ComponentState", "Element", "Workflow", "Intent"]

_OPS_EXACT = [("Agent", "id"), ("Task", "id"), ("Claim", "target_gid"), ("Branch", "name"), ("Commit", "id"),
              ("WorkItem", "gid"), ("WorkItem", "scope")]
_PROFILE_EXACT = [("Preference", "key"), ("Memory", "gid"), ("Session", "id"), ("Request", "gid"),
                  ("SiteEntry", "gid"), ("ExplorationPolicy", "name")]
_PROFILE_VECTOR = ["Memory", "Request"]


def _existing(store: GraphStore, key: str) -> set[tuple[str, str]]:
    try:
        rows = store.ro_query(key, "CALL db.indexes() YIELD label, properties, types RETURN label, properties, types")
    except Exception:
        return set()
    out = set()
    for label, props, types in rows:
        for p in props:
            for t in (types.get(p) if isinstance(types, dict) else types) or []:
                out.add((label, f"{p}:{str(t).upper()}"))
    return out


def _run(store: GraphStore, key: str, cypher: str) -> None:
    try:
        store.query(key, cypher)
    except Exception as e:  # concurrent creation or already-present index
        if "already" not in str(e).lower() and "exist" not in str(e).lower():
            raise


def _exact(store, key, have, label, prop):
    if (label, f"{prop}:RANGE") not in have:
        _run(store, key, f"CREATE INDEX FOR (n:{label}) ON (n.{prop})")


def _vector(store, key, have, label, dim):
    if (label, "emb:VECTOR") not in have:
        _run(store, key, f"CREATE VECTOR INDEX FOR (n:{label}) ON (n.emb) "
                         f"OPTIONS {{dimension:{dim}, similarityFunction:'cosine'}}")


def _unique(store, key, label):
    try:
        store.execute("GRAPH.CONSTRAINT", "CREATE", key, "UNIQUE", "NODE", label, "PROPERTIES", "1", "gid")
    except Exception as e:
        if "already" not in str(e).lower() and "exist" not in str(e).lower():
            raise


def ensure_device_schema(store: GraphStore, key: str, dim: int = 384) -> None:
    store.query(key, "RETURN 1")  # creates the graph key if missing
    have = _existing(store, key)
    for label in _DEVICE_EXACT:
        _exact(store, key, have, label, "gid")
    for rel, prop in _DEVICE_REL:
        if (rel, f"{prop}:RANGE") not in have:
            _run(store, key, f"CREATE INDEX FOR ()-[r:{rel}]-() ON (r.{prop})")
    for label, prop in _DEVICE_FULLTEXT:
        if (label, f"{prop}:FULLTEXT") not in have:
            _run(store, key, f"CALL db.idx.fulltext.createNodeIndex('{label}', '{prop}')")
    for label in _DEVICE_VECTOR:
        _vector(store, key, have, label, dim)
    for label in _DEVICE_UNIQUE:
        _unique(store, key, label)


def ensure_ops_schema(store: GraphStore, key: str) -> None:
    store.query(key, "RETURN 1")
    have = _existing(store, key)
    for label, prop in _OPS_EXACT:
        _exact(store, key, have, label, prop)


def ensure_profile_schema(store: GraphStore, key: str, dim: int = 384) -> None:
    store.query(key, "RETURN 1")
    have = _existing(store, key)
    for label, prop in _PROFILE_EXACT:
        _exact(store, key, have, label, prop)
    for label in _PROFILE_VECTOR:
        _vector(store, key, have, label, dim)
