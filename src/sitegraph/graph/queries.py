from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime

from ..db import GraphStore
from ..embed import cosine


@dataclass
class Target:
    kind: str
    gid: str
    text: str
    score: float
    item: str | None = None   # set when the target was found through a list item the request names ('Travel')
    intent: str | None = None  # the Intent that led to it (its hits are counted when the goal is reached)


@dataclass
class RouteStep:
    action_gid: str
    label: str
    el_gid: str
    src: str
    dst: str
    reliability: float = 0.5
    last_seen: int | None = None
    pre_cstates: list = field(default_factory=list)
    requires_inputs: list = field(default_factory=list)
    value_var: str | None = None
    action_type: str = "click"
    risk_cost: int = 0


@dataclass
class Route:
    steps: list[RouteStep]
    weight: float
    risk_cost: int
    probability: float


ITEM_TARGET_SCORE = 0.9


def _targets_from_list_items(store: GraphStore, key: str, text: str) -> list[Target]:
    """Screens reached by clicking a list item the request names (e.g. 'Mystery' in 'take me to the mystery
    category' -> the shared category screen), even if that exact instance was never visited."""
    from ..browser.items import pick_by_text

    out: list[Target] = []
    rows = store.ro_query(key, "MATCH (e:Element {is_template:true}) WHERE e.sample_items IS NOT NULL "
                               "RETURN e.gid, e.sample_items")
    for el_gid, items in rows:
        item = pick_by_text(list(items), text)
        if item is None:
            continue
        for g, summary in store.ro_query(key, "MATCH (:State)-[r:ACTION {el_gid:$e}]->(t:State) "
                                              "RETURN DISTINCT t.gid, coalesce(t.summary, '')", {"e": el_gid}):
            out.append(Target("State", g, summary, ITEM_TARGET_SCORE, item))
    return out


def find_targets(store: GraphStore, key: str, vec: list[float], k: int = 5, text: str | None = None) -> list[Target]:
    out: dict[str, Target] = {}
    if text:
        for t in _targets_from_list_items(store, key, text):
            out[t.gid] = t
    rows = store.ro_query(key, """
        CALL db.idx.vector.queryNodes('Intent', 'emb', $k, vecf32($v)) YIELD node
        MATCH (node)-[f:FULFILLED_BY]->(t)
        RETURN labels(t)[0], t.gid, coalesce(t.summary, t.name, node.text), node.emb, coalesce(f.score, 1.0),
               node.gid""",
                          {"k": k, "v": vec})
    for label, g, text, emb, fscore, ig in rows:
        score = cosine(vec, list(emb)) * fscore
        if g not in out or out[g].score < score:
            out[g] = Target(label, g, text or "", score, out[g].item if g in out else None, ig)
    for label in ("Workflow", "State"):
        for g, text, sim in find_nodes(store, key, label, vec, k, "summary" if label == "State" else "name"):
            if g not in out or out[g].score < sim:
                out[g] = Target(label, g, text, sim, out[g].item if g in out else None,
                                out[g].intent if g in out else None)
    return sorted(out.values(), key=lambda t: t.score, reverse=True)[:k]


def find_nodes(store: GraphStore, key: str, label: str, vec: list[float], k: int = 5,
               text_prop: str = "name") -> list[tuple[str, str, float]]:
    rows = store.ro_query(key, f"CALL db.idx.vector.queryNodes('{label}', 'emb', $k, vecf32($v)) YIELD node "
                               f"RETURN node.gid, coalesce(node.{text_prop}, ''), node.emb", {"k": k, "v": vec})
    return sorted(((g, t, cosine(vec, list(e))) for g, t, e in rows), key=lambda r: r[2], reverse=True)


_ROUTE = """
MATCH (s:State {gid:$src}), (t:State {gid:$dst})
CALL algo.SPpaths({sourceNode: s, targetNode: t, relTypes: ['ACTION'], weightProp: 'cost',
                   costProp: 'risk_cost', maxCost: $max_cost, maxLen: $max_len, pathCount: $k})
YIELD path, pathWeight, pathCost
RETURN [n IN nodes(path) | n.gid],
       [r IN relationships(path) | [r.gid, r.label, r.el_gid, r.reliability, r.last_seen, r.pre_cstates,
                                    r.requires_inputs, r.value_var, r.action_type, r.risk_cost]],
       pathWeight, pathCost
ORDER BY pathWeight"""


def best_routes(store: GraphStore, key: str, src: str, dst: str, max_risk_cost: int, k: int = 3,
                max_len: int = 30) -> list[Route]:
    if src == dst:
        return [Route([], 0.0, 0, 1.0)]
    rows = store.ro_query(key, _ROUTE, {"src": src, "dst": dst, "max_cost": int(max_risk_cost),
                                        "max_len": max_len, "k": k})
    routes = []
    for ns, rs, weight, cost in rows:
        steps = [RouteStep(action_gid=r[0], label=r[1] or "", el_gid=r[2], src=ns[i], dst=ns[i + 1],
                           reliability=r[3] if r[3] is not None else 0.5, last_seen=r[4], pre_cstates=r[5] or [],
                           requires_inputs=r[6] or [], value_var=r[7], action_type=r[8] or "click",
                           risk_cost=r[9] or 0) for i, r in enumerate(rs)]
        routes.append(Route(steps, float(weight), int(cost or 0), math.prod(s.reliability for s in steps)))
    return routes


def explain(route: Route) -> str:
    lines = []
    for i, s in enumerate(route.steps, 1):
        seen = datetime.fromtimestamp(s.last_seen / 1000).strftime("%Y-%m-%d %H:%M") if s.last_seen else "never"
        lines.append(f"{i}. {s.label} (reliability {s.reliability:.2f}, last seen {seen})")
    lines.append(f"Estimated success: {route.probability:.0%} over {len(route.steps)} step(s)")
    return "\n".join(lines)


def next_actions(store: GraphStore, key: str, state: str, max_risk_cost: int, k: int = 5) -> list[dict]:
    rows = store.ro_query(key, """
        MATCH (s:State {gid:$g})-[a:ACTION]->(t:State) WHERE a.risk_cost <= $mrc
        RETURN a.gid, a.label, t.gid, coalesce(t.summary, ''), a.seen_count, a.reliability,
               a.seen_count * a.reliability AS score
        ORDER BY score DESC LIMIT $k""", {"g": state, "mrc": max_risk_cost, "k": k})
    return [dict(action_gid=r[0], label=r[1], target=r[2], target_summary=r[3], seen=r[4], reliability=r[5],
                 score=r[6]) for r in rows]


def frontier_states(store: GraphStore, key: str, depth: int = 8, limit: int = 10) -> list[dict]:
    rows = store.ro_query(key, """
        MATCH (e:State:Entry)
        CALL algo.bfs(e, $depth, 'ACTION') YIELD nodes
        UNWIND nodes + [e] AS n
        WITH DISTINCT n WHERE n:Frontier
        RETURN n.gid, coalesce(n.summary, ''), n.untried_count, coalesce(n.harmonic, 0.0)
        ORDER BY n.untried_count DESC, coalesce(n.harmonic, 0.0) DESC LIMIT $limit""", {"depth": depth, "limit": limit})
    return [dict(gid=r[0], summary=r[1], untried=r[2], harmonic=r[3]) for r in rows]


def mine_sequences(store: GraphStore, key: str, min_freq: int = 2, limit: int = 10) -> list[dict]:
    rows = store.ro_query(key, """
        MATCH (a:Step)-[:NEXT]->(b:Step)
        WHERE a.ok AND b.ok AND a.action_gid IS NOT NULL AND b.action_gid IS NOT NULL
          AND a.action_gid <> b.action_gid
        RETURN a.action_gid, b.action_gid, count(*) AS freq ORDER BY freq DESC LIMIT 200""")
    pairs = {(a, b): f for a, b, f in rows if f >= min_freq}
    out = []
    for (a, b), f in sorted(pairs.items(), key=lambda kv: kv[1], reverse=True):
        seq = [a, b]
        while True:  # greedy extension with the most frequent follower that keeps frequency >= min_freq
            nxt = [(y, fy) for (x, y), fy in pairs.items() if x == seq[-1] and y not in seq]
            if not nxt:
                break
            seq.append(max(nxt, key=lambda t: t[1])[0])
        if not any(seq == o["actions"][:len(seq)] for o in out):
            out.append({"actions": seq, "freq": f})
        if len(out) >= limit:
            break
    return out


def impact_of_component(store: GraphStore, key: str, comp_gid: str) -> list[dict]:
    rows = store.ro_query(key, """
        MATCH (c:Component {gid:$c})-[:HAS_CSTATE]->(:ComponentState)-[:CONTAINS]->(e:Element)
        MATCH (ws:WStep)-[:ON]->(e), (ws)-[:STEP_OF]->(w:Workflow)
        OPTIONAL MATCH (sb:Storyboard)-[:BASED_ON]->(w)
        RETURN e.name_sample, collect(DISTINCT w.name), collect(DISTINCT sb.title)""", {"c": comp_gid})
    return [dict(element=r[0], workflows=r[1], demos=r[2]) for r in rows]


_EL_FIELDS = ["gid", "role", "name_sample", "name_norm", "locators", "fingerprint", "is_template", "risk",
              "risk_cost", "input_kind", "in_form", "href_template", "item_locator"]


def load_element(store: GraphStore, key: str, el_gid: str) -> dict | None:
    rows = store.ro_query(key, "MATCH (x:Element {gid:$g}) RETURN " + ", ".join(f"x.{f}" for f in _EL_FIELDS),
                          {"g": el_gid})
    if not rows:
        return None
    d = dict(zip(_EL_FIELDS, rows[0]))
    if isinstance(d.get("fingerprint"), str):
        d["fingerprint"] = json.loads(d["fingerprint"])
    return d


def entry_state(store: GraphStore, key: str) -> tuple[str, str] | None:
    rows = store.ro_query(key, "MATCH (p:Page)-[:HAS_STATE]->(s:State:Entry) RETURN s.gid, p.example_urls[0] LIMIT 1")
    return (rows[0][0], rows[0][1]) if rows else None


def state_url(store: GraphStore, key: str, state_gid: str) -> str | None:
    rows = store.ro_query(key, "MATCH (p:Page)-[:HAS_STATE]->(s:State {gid:$g}) RETURN p.example_urls[0]",
                          {"g": state_gid})
    return rows[0][0] if rows else None


def recommendations(store: GraphStore, key: str, max_risk_cost: int) -> dict:
    """Next actions from the entry screen, exploration frontier, suggested workflows and demo candidates."""
    entry = entry_state(store, key)
    nexts = next_actions(store, key, entry[0], max_risk_cost) if entry else []
    demo = store.ro_query(key, """
        MATCH (s:State) WHERE s.demo_score IS NOT NULL
        RETURN s.gid, coalesce(s.summary, ''), s.demo_score ORDER BY s.demo_score DESC LIMIT 5""")
    wfs = store.ro_query(key, """
        MATCH (w:Workflow) RETURN w.gid, w.name, coalesce(w.demo_score, 0.0), coalesce(w.run_count, 0)
        ORDER BY coalesce(w.demo_score, 0.0) DESC LIMIT 5""")
    return {"next_actions": nexts,
            "frontier": frontier_states(store, key),
            "suggested_workflows": mine_sequences(store, key),
            "demo_candidates": [{"gid": g, "summary": s, "score": d} for g, s, d in demo],
            "workflows": [{"gid": g, "name": n, "demo_score": d, "runs": r} for g, n, d, r in wfs]}


def fulltext_elements(store: GraphStore, key: str, text: str, role: str | None, exclude_gid: str | None,
                      k: int = 5, max_risk_cost: int | None = None) -> list[dict]:
    """Other known elements whose name matches `text` (graph healing alternatives). With `max_risk_cost`, only
    elements no riskier than that: a broken 'Cart' must never be healed into 'Empty cart'."""
    words = " ".join(re.findall(r"[A-Za-z0-9]+", text or ""))
    if not words:
        return []
    rows = store.ro_query(key, "CALL db.idx.fulltext.queryNodes('Element', $q) YIELD node "
                               "RETURN node.gid, node.role, coalesce(node.risk_cost, 0) LIMIT 25", {"q": words})
    gids = [g for g, r, rc in rows if (role is None or r == role) and g != exclude_gid
            and (max_risk_cost is None or rc <= max_risk_cost)][:k]
    return [e for e in (load_element(store, key, g) for g in gids) if e]
