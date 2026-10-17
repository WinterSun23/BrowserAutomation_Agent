"""What the semantic graph is for: carry a workflow to another device, steer exploration of a new site towards the kinds
of screens that matter, and report where devices differ."""
from __future__ import annotations

import json

from ..agents.workflows import load_workflow_steps, save_workflow_from_actions
from ..db import device_key
from ..embed import cosine
from ..graph.queries import best_routes, load_element
from .distiller import semantic_key


class PortError(Exception):
    pass


def _map_state(store, uid: str, from_key: str, state_gid: str, to_key: str) -> str | None:
    rows = store.ro_query(semantic_key(uid), """
        MATCH (l:LogicalState)-[:REALISED_AS]->(:DeviceStateRef {ref_graph:$fk, ref_gid:$s})
        MATCH (l)-[:REALISED_AS]->(d:DeviceStateRef {ref_graph:$tk}) RETURN d.ref_gid LIMIT 1""",
                          {"fk": from_key, "s": state_gid, "tk": to_key})
    return rows[0][0] if rows else None


def port_workflow(store, uid: str, embedder, domain: str, wf_gid: str, from_device: str, to_device: str) -> str:
    """Re-create a workflow on another device: each step's screens are mapped through the LogicalStates; the action
    between them is the most similar one on the target device, or a short route if there is no direct one."""
    fk, tk = device_key(uid, domain, from_device), device_key(uid, domain, to_device)
    steps = [s for s in load_workflow_steps(store, fk, wf_gid) if s["kind"] == "ACTION"]
    if not steps:
        raise PortError("workflow has no actions to port")
    actions: list[str] = []
    for i, st in enumerate(steps):
        src, dst = _map_state(store, uid, fk, st["src"], tk), _map_state(store, uid, fk, st["dst"], tk)
        if not src or not dst:
            raise PortError(f"step {i}: no equivalent screen on {to_device}; explore it first")
        old_el = load_element(store, fk, st["el_gid"]) or {}
        direct = store.ro_query(tk, """MATCH (:State {gid:$a})-[r:ACTION]->(:State {gid:$b})
                                       MATCH (e:Element) WHERE e.gid = r.el_gid
                                       RETURN r.gid, e.name_sample, e.role ORDER BY r.reliability DESC""",
                                {"a": src, "b": dst})
        if direct:
            want = embedder.embed([f"{old_el.get('role', '')} {old_el.get('name_sample', '')}"])[0]
            best = max(direct, key=lambda r: cosine(want, embedder.embed([f"{r[2]} {r[1]}"])[0]))
            actions.append(best[0])
            continue
        routes = best_routes(store, tk, src, dst, 1)
        if not routes:
            raise PortError(f"step {i}: no route between equivalent screens on {to_device}")
        actions.extend(s.action_gid for s in routes[0].steps)
    name, goal = store.ro_query(fk, "MATCH (w:Workflow {gid:$w}) RETURN w.name, w.goal", {"w": wf_gid})[0]
    new = save_workflow_from_actions(store, tk, embedder, actions, f"{name} ({to_device})", goal)
    store.query(tk, "MATCH (w:Workflow {gid:$w}) SET w.source = 'ported', w.ported_from = $f",
                {"w": new, "f": f"{fk}|{wf_gid}"})
    return new


def exploration_priority(store, uid: str, embedder):
    """Higher score for elements that look like steps of frequent flow patterns (guided exploration on new sites)."""
    rows = store.ro_query(semantic_key(uid), """
        MATCH (p:StepPattern) OPTIONAL MATCH (f:FlowPattern)-[:STEP]->(p)
        RETURN p.emb, coalesce(max(f.support), 0)""")
    pats = [(list(e), s) for e, s in rows if e]

    def score(el) -> float:
        if not pats:
            return 0.0
        v = embedder.embed([el.name_sample or el.name_norm])[0]
        return float(max(cosine(v, e) * (1 + s) for e, s in pats))

    return score


def parity_report(store, uid: str) -> list[dict]:
    """Screens that exist on some devices but not others, or take more steps to reach on a device than on desktop."""
    sk = semantic_key(uid)
    devices = {r[0] for r in store.ro_query(sk, "MATCH ()-[r:REALISED_AS]->() RETURN DISTINCT r.device")}
    out = []
    for lg, summary, devs, steps in store.ro_query(sk, "MATCH (l:LogicalState) RETURN l.gid, l.summary, l.devices, "
                                                       "l.device_steps"):
        missing = sorted(devices - set(devs or []))
        deltas = json.loads(steps) if steps else {}
        extra = {d: n - deltas.get("desktop", n) for d, n in deltas.items() if d != "desktop"}
        if missing or any(v > 0 for v in extra.values()):
            out.append({"logical_state": lg, "summary": summary, "missing_on": missing, "extra_steps": extra})
    return out


def pattern_of_component(store, uid: str, graph: str, comp_gid: str) -> str | None:
    rows = store.ro_query(semantic_key(uid), "MATCH (:ComponentRef {ref_graph:$g, ref_gid:$c})-[:INSTANCE_OF]->"
                                             "(u:UIPattern) RETURN u.gid LIMIT 1", {"g": graph, "c": comp_gid})
    return rows[0][0] if rows else None
