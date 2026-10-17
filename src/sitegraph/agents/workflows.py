from __future__ import annotations

from ..ids import gid, now_ms
from ..normalize import normalize_text


def _edge(store, key, action_gid: str) -> dict | None:
    for kind in ("ACTION", "TRANSITION"):
        rows = store.ro_query(key, f"MATCH (a)-[r:{kind} {{gid:$g}}]->(b) RETURN r.gid, a.gid, b.gid, r.el_gid, "
                                   "r.label, r.reliability, r.pre_cstates, r.requires_inputs, r.value_var, "
                                   "r.action_type", {"g": action_gid})
        if rows:
            r = rows[0]
            return dict(kind=kind, action_gid=r[0], src=r[1], dst=r[2], el_gid=r[3], label=r[4] or "",
                        reliability=r[5] if r[5] is not None else 0.5, pre_cstates=r[6] or [],
                        requires_inputs=r[7] or [], value_var=r[8], action_type=r[9] or "click")
    return None


def save_workflow(store, key, embedder, steps: list[tuple], name: str, goal: str, source: str = "user",
                  calls: list[str] | None = None) -> str:
    """steps are (action_gid, value_var, from_gid); `calls` are sub-workflows run after them (composition)."""
    action_gids = [s[0] for s in steps]
    params = sorted({v for s in steps for v in (s[1] or "").split(",") if v})
    wf = gid("wf", *(action_gids or []), *(calls or []), *params)
    v = embedder.embed([f"{name}. {goal}"])[0]
    store.query(key, """MERGE (w:Workflow {gid:$wf})
        SET w.first_seen = coalesce(w.first_seen, $now), w.run_count = coalesce(w.run_count, 0),
            w.success_rate = coalesce(w.success_rate, 0.0), w.version = coalesce(w.version, 1),
            w.name = $name, w.goal = $goal, w.emb = vecf32($v), w.params = $params, w.source = $src,
            w.status = 'active'""",
                {"wf": wf, "now": now_ms(), "name": name, "goal": goal, "v": v, "params": params, "src": source})
    for order, (ag, var, from_gid) in enumerate(steps):
        e = _edge(store, key, ag) or {}
        ws = gid("ws", wf, order)
        store.query(key, "MERGE (s:WStep {gid:$ws}) SET s.order = $o, s.action_gid = $ag, s.value_var = $var "
                         "WITH s MATCH (w:Workflow {gid:$wf}) MERGE (s)-[:STEP_OF]->(w)",
                    {"ws": ws, "o": order, "ag": ag, "var": var, "wf": wf})
        src = from_gid or e.get("src")
        if src:
            store.query(key, "MATCH (s:WStep {gid:$ws}), (f:State {gid:$f}) MERGE (s)-[:FROM]->(f)", {"ws": ws, "f": src})
        if e.get("el_gid"):
            store.query(key, "MATCH (s:WStep {gid:$ws}), (x:Element {gid:$e}) MERGE (s)-[:ON]->(x)",
                        {"ws": ws, "e": e["el_gid"]})
    for order, sub in enumerate(calls or []):
        ws = gid("ws", wf, "call", order)
        store.query(key, """MERGE (s:WStep {gid:$ws}) SET s.order = $o, s.kind = 'call'
                            WITH s MATCH (w:Workflow {gid:$wf}), (x:Workflow {gid:$sub})
                            MERGE (s)-[:STEP_OF]->(w) MERGE (s)-[:CALLS]->(x)""",
                    {"ws": ws, "o": order, "wf": wf, "sub": sub})
    ig = gid("in", normalize_text(goal))
    store.query(key, """MERGE (i:Intent {gid:$ig}) SET i.text = coalesce(i.text, $t), i.emb = vecf32($v),
                        i.source = coalesce(i.source, 'user'), i.hits = coalesce(i.hits, 0)
                        WITH i MATCH (w:Workflow {gid:$wf}) MERGE (i)-[f:FULFILLED_BY]->(w) SET f.score = 1.0""",
                {"ig": ig, "t": goal, "v": embedder.embed([goal])[0], "wf": wf})
    return wf


def save_workflow_from_run(store, key, embedder, run_gid: str, name: str, goal: str) -> str:
    rows = store.ro_query(key, """
        MATCH (st:Step)-[:IN_RUN]->(:Run {gid:$r}) WHERE st.ok AND st.action_gid IS NOT NULL
        OPTIONAL MATCH (st)-[:FROM]->(f:State)
        RETURN st.action_gid, st.value_var, f.gid ORDER BY st.i""", {"r": run_gid})
    if not rows:
        raise ValueError(f"run {run_gid} has no successful steps to save")
    return save_workflow(store, key, embedder, [tuple(r) for r in rows], name, goal)


def save_workflow_from_actions(store, key, embedder, action_gids: list[str], name: str, goal: str) -> str:
    """Save a sequence of known actions (for example one mined from past runs) as a workflow."""
    steps = []
    for ag in action_gids:
        e = _edge(store, key, ag)
        if e is None:
            raise ValueError(f"unknown action {ag}")
        steps.append((ag, e["value_var"], e["src"] if e["kind"] == "ACTION" else None))
    return save_workflow(store, key, embedder, steps, name, goal, source="mined")


def compose_workflow(store, key, embedder, sub_wf_gids: list[str], name: str, goal: str) -> str:
    """A workflow that runs other workflows in order (CALLS)."""
    return save_workflow(store, key, embedder, [], name, goal, source="composed", calls=sub_wf_gids)


def load_workflow_steps(store, key, wf_gid: str, _depth: int = 0) -> list[dict]:
    """The flat list of actions of a workflow; CALLS to sub-workflows are expanded (up to 3 levels)."""
    rows = store.ro_query(key, """MATCH (s:WStep)-[:STEP_OF]->(:Workflow {gid:$w})
                                  OPTIONAL MATCH (s)-[:CALLS]->(sub:Workflow)
                                  RETURN s.order, s.action_gid, sub.gid ORDER BY s.order""", {"w": wf_gid})
    out = []
    for order, ag, sub in rows:
        if sub and _depth < 3:
            out.extend(load_workflow_steps(store, key, sub, _depth + 1))
        elif ag:
            e = _edge(store, key, ag)
            if e:
                out.append({"order": order, **e})
    for i, s in enumerate(out):
        s["order"] = i
    return out
