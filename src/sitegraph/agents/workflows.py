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


def save_workflow_from_run(store, key, embedder, run_gid: str, name: str, goal: str) -> str:
    rows = store.ro_query(key, """
        MATCH (st:Step)-[:IN_RUN]->(:Run {gid:$r}) WHERE st.ok AND st.action_gid IS NOT NULL
        OPTIONAL MATCH (st)-[:FROM]->(f:State)
        RETURN st.i, st.action_gid, st.value_var, f.gid ORDER BY st.i""", {"r": run_gid})
    if not rows:
        raise ValueError(f"run {run_gid} has no successful steps to save")
    action_gids = [r[1] for r in rows]
    params = sorted({v for r in rows for v in (r[2] or "").split(",") if v})
    wf = gid("wf", *action_gids, *params)
    v = embedder.embed([f"{name}. {goal}"])[0]
    store.query(key, """MERGE (w:Workflow {gid:$wf})
        SET w.first_seen = coalesce(w.first_seen, $now), w.run_count = coalesce(w.run_count, 0),
            w.success_rate = coalesce(w.success_rate, 0.0), w.version = coalesce(w.version, 1),
            w.name = $name, w.goal = $goal, w.emb = vecf32($v), w.params = $params, w.source = 'user',
            w.status = 'active'""", {"wf": wf, "now": now_ms(), "name": name, "goal": goal, "v": v, "params": params})
    for order, (_, ag, var, from_gid) in enumerate(rows):
        e = _edge(store, key, ag) or {}
        ws = gid("ws", wf, order)
        store.query(key, "MERGE (s:WStep {gid:$ws}) SET s.order = $o, s.action_gid = $ag, s.value_var = $var "
                         "WITH s MATCH (w:Workflow {gid:$wf}) MERGE (s)-[:STEP_OF]->(w)",
                    {"ws": ws, "o": order, "ag": ag, "var": var, "wf": wf})
        if from_gid:
            store.query(key, "MATCH (s:WStep {gid:$ws}), (f:State {gid:$f}) MERGE (s)-[:FROM]->(f)",
                        {"ws": ws, "f": from_gid})
        if e.get("el_gid"):
            store.query(key, "MATCH (s:WStep {gid:$ws}), (x:Element {gid:$e}) MERGE (s)-[:ON]->(x)",
                        {"ws": ws, "e": e["el_gid"]})
    ig = gid("in", normalize_text(goal))
    gv = embedder.embed([goal])[0]
    store.query(key, """MERGE (i:Intent {gid:$ig}) SET i.text = coalesce(i.text, $t), i.emb = vecf32($v),
                        i.source = coalesce(i.source, 'user'), i.hits = coalesce(i.hits, 0)
                        WITH i MATCH (w:Workflow {gid:$wf}) MERGE (i)-[f:FULFILLED_BY]->(w) SET f.score = 1.0""",
                {"ig": ig, "t": goal, "v": gv, "wf": wf})
    return wf


def load_workflow_steps(store, key, wf_gid: str) -> list[dict]:
    rows = store.ro_query(key, "MATCH (s:WStep)-[:STEP_OF]->(:Workflow {gid:$w}) RETURN s.order, s.action_gid "
                               "ORDER BY s.order", {"w": wf_gid})
    out = []
    for order, ag in rows:
        e = _edge(store, key, ag)
        if e:
            out.append({"order": order, **e})
    return out
