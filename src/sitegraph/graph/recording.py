"""Recorded map operations: every MapWriter operation is logged as a `Change` with the prior state of what it touched,
so a branch can be merged by replaying its changes and a commit can be reverted from the snapshots (spec §6.1-6.3)."""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass

from ..capture.model import obs_from_dict, obs_to_dict
from ..ids import gid, now_ms
from ..normalize import normalize_text
from .map_writer import _BROKEN, MapWriter

STRIP_ACTIVITY = ("MATCH (n) WHERE n:Run OR n:Step OR n:Observation OR n:DemoRequest OR n:Storyboard OR n:Shot "
                  "OR n:Render DETACH DELETE n")
_EL_KEYS = ("gid", "name_sample", "name_norm", "risk", "risk_cost")


def _el(el: dict) -> dict:
    return {k: el.get(k) for k in _EL_KEYS}


@dataclass
class Change:
    seq: int
    op: str
    args: dict
    touched: list
    ts: int

    def to_props(self) -> dict:
        return {"seq": self.seq, "op": self.op, "args": json.dumps(self.args, sort_keys=True, default=str),
                "touched": json.dumps(self.touched, sort_keys=True, default=str), "ts": self.ts}

    @classmethod
    def from_props(cls, p: dict) -> "Change":
        return cls(seq=p["seq"], op=p["op"], args=json.loads(p["args"]), touched=json.loads(p["touched"]), ts=p["ts"])

    def digest(self) -> str:
        return hashlib.sha1(f"{self.op}|{json.dumps(self.args, sort_keys=True, default=str)}".encode()).hexdigest()


def replay(writer: MapWriter, ch: Change):
    args = dict(ch.args)
    if ch.op == "write_state":
        args["obs"] = obs_from_dict(args["obs"])
    return getattr(writer, ch.op)(**args)


class RecordingMapWriter(MapWriter):
    """MapWriter that logs every map-fact operation with prior-state snapshots (for merge replay and revert)."""

    def __init__(self, store, key, settings, now=now_ms):
        super().__init__(store, key, settings, now)
        self.changes: list[Change] = []
        self._inner = 0

    @contextmanager
    def _nested(self):
        """Operations an operation is built from (quarantine = delete_edge + restore_edge) are not logged separately."""
        self._inner += 1
        try:
            yield
        finally:
            self._inner -= 1

    def _log(self, op: str, args: dict, touched: list) -> None:
        if self._inner:
            return
        self.changes.append(Change(seq=len(self.changes), op=op, args=args, touched=touched, ts=self.now()))

    def _nodes(self, label: str, gids) -> list:
        return [{"kind": "node", "label": label, "gid": g, "prev": p} for g, p in self.node_props(label, gids).items()]

    def _edge(self, rtype: str, edge_gid: str) -> list:
        return [{"kind": "edge", "type": rtype, "gid": edge_gid, "prev": self.edge_props(rtype, edge_gid)}]

    def _rels(self, label: str, node_gid: str) -> list:
        """Incident relationships without their own gid (structural edges), so a revert can rebuild them."""
        out = self.ro(f"MATCH (n:{label} {{gid:$g}})-[r]->(m) RETURN 'out', type(r), labels(m)[0], m.gid, "
                      "properties(r)", g=node_gid)
        inc = self.ro(f"MATCH (n:{label} {{gid:$g}})<-[r]-(m) RETURN 'in', type(r), labels(m)[0], m.gid, "
                      "properties(r)", g=node_gid)
        return [list(r) for r in out + inc if "gid" not in (r[4] or {})]

    # ---------- map facts ----------
    def write_state(self, obs, screenshot_ref=None, snapshot_ref=None, phash=None):
        sg = self.resolve_state(obs) or obs.state_gid
        touched = (self._nodes("Page", [obs.page_gid]) + self._nodes("State", [sg])
                   + self._nodes("Component", {c.gid for c in obs.components})
                   + self._nodes("ComponentState", {c.cs_gid for c in obs.components})
                   + self._nodes("Element", {e.gid for e in obs.elements()}))
        out = super().write_state(obs, screenshot_ref, snapshot_ref, phash)
        self._log("write_state", {"obs": obs_to_dict(obs), "screenshot_ref": screenshot_ref,
                                  "snapshot_ref": snapshot_ref, "phash": phash}, touched)
        return out

    def write_action(self, src, dst, el, action_type="click", value_var=None, pre_cstates=(), requires_inputs=(),
                     success=True):
        touched = self._edge("ACTION", self.action_gid(src, el["gid"], action_type, value_var, dst))
        out = super().write_action(src, dst, el, action_type, value_var, pre_cstates, requires_inputs, success)
        self._log("write_action", {"src": src, "dst": dst, "el": _el(el), "action_type": action_type,
                                   "value_var": value_var, "pre_cstates": list(pre_cstates),
                                   "requires_inputs": list(requires_inputs), "success": success}, touched)
        return out

    def record_attempt(self, action_gid, success):
        touched = self._edge("ACTION", action_gid)
        super().record_attempt(action_gid, success)
        self._log("record_attempt", {"action_gid": action_gid, "success": success}, touched)

    def add_pre_actions(self, action_gid, actions):
        touched = self._edge("ACTION", action_gid)
        super().add_pre_actions(action_gid, actions)
        self._log("add_pre_actions", {"action_gid": action_gid, "actions": list(actions)}, touched)

    def write_transition(self, comp_gid, src_cs, dst_cs, el, action_type="click", src_name="absent", dst_name="open"):
        touched = (self._nodes("Component", [comp_gid]) + self._nodes("ComponentState", [src_cs, dst_cs])
                   + self._edge("TRANSITION", self.transition_gid(src_cs, el["gid"], action_type, dst_cs)))
        out = super().write_transition(comp_gid, src_cs, dst_cs, el, action_type, src_name, dst_name)
        self._log("write_transition", {"comp_gid": comp_gid, "src_cs": src_cs, "dst_cs": dst_cs, "el": _el(el),
                                       "action_type": action_type, "src_name": src_name, "dst_name": dst_name}, touched)
        return out

    def mark_tried(self, el_gid):
        touched = self._nodes("Element", [el_gid])
        super().mark_tried(el_gid)
        self._log("mark_tried", {"el_gid": el_gid}, touched)

    def set_entry(self, state_gid):
        touched = self._nodes("State", [state_gid])
        super().set_entry(state_gid)
        self._log("set_entry", {"state_gid": state_gid}, touched)

    def set_cstate_shot(self, cs_gid, ref):
        touched = self._nodes("ComponentState", [cs_gid])
        super().set_cstate_shot(cs_gid, ref)
        self._log("set_cstate_shot", {"cs_gid": cs_gid, "ref": ref}, touched)

    def set_cstate_snapshot(self, cs_gid, ref):
        touched = self._nodes("ComponentState", [cs_gid])
        super().set_cstate_snapshot(cs_gid, ref)
        self._log("set_cstate_snapshot", {"cs_gid": cs_gid, "ref": ref}, touched)

    def bump_locator(self, el_gid, locator, healed=False):
        touched = self._nodes("Element", [el_gid])
        super().bump_locator(el_gid, locator, healed)
        self._log("bump_locator", {"el_gid": el_gid, "locator": locator, "healed": healed}, touched)

    # ---------- curator edits and generic primitives ----------
    def merge_states(self, keep, merge):
        touched = self._nodes("State", [keep, merge])
        if touched[1]["prev"] is not None:
            touched[1]["prev"]["_rels"] = self._rels("State", merge)
            touched[1]["prev"]["_labels"] = self.ro("MATCH (s:State {gid:$g}) RETURN labels(s)", g=merge)[0][0]
        for e in self.action_edges_of(merge):
            touched += self._edge("ACTION", e["gid"])
            new_gid = self.merge_target_gid(e, keep, merge)
            if new_gid:
                touched += self._edge("ACTION", new_gid)
        out = super().merge_states(keep, merge)
        self._log("merge_states", {"keep": keep, "merge": merge}, touched)
        return out

    def rename_state(self, state_gid, summary, titles, state_vec=None, intents=None):
        touched = (self._nodes("State", [state_gid])
                   + self._nodes("Intent", [gid("in", normalize_text(t)) for t, _ in intents or []]))
        out = super().rename_state(state_gid, summary, titles, state_vec, intents)
        self._log("rename_state", {"state_gid": state_gid, "summary": summary, "titles": list(titles),
                                   "state_vec": state_vec, "intents": [list(i) for i in intents or []]}, touched)
        return out

    def quarantine(self, rtype, edge_gid, reason, superseded_by=None):
        touched = self._edge(rtype, edge_gid) + self._edge(_BROKEN[rtype], edge_gid)
        with self._nested():
            ok = super().quarantine(rtype, edge_gid, reason, superseded_by)
        if ok:
            self._log("quarantine", {"rtype": rtype, "edge_gid": edge_gid, "reason": reason,
                                     "superseded_by": superseded_by}, touched)
        return ok

    def unquarantine(self, rtype, edge_gid):
        touched = self._edge(rtype, edge_gid) + self._edge(_BROKEN[rtype], edge_gid)
        with self._nested():
            ok = super().unquarantine(rtype, edge_gid)
        if ok:
            self._log("unquarantine", {"rtype": rtype, "edge_gid": edge_gid}, touched)
        return ok

    def set_props(self, label, node_gid, props):
        touched = self._nodes(label, [node_gid])
        super().set_props(label, node_gid, props)
        self._log("set_props", {"label": label, "node_gid": node_gid, "props": props}, touched)

    def restore_node(self, label, node_gid, props):
        touched = self._nodes(label, [node_gid])
        super().restore_node(label, node_gid, props)
        self._log("restore_node", {"label": label, "node_gid": node_gid, "props": props}, touched)

    def delete_node(self, label, node_gid):
        touched = self._nodes(label, [node_gid])
        if touched and touched[0]["prev"] is not None:
            touched[0]["prev"]["_rels"] = self._rels(label, node_gid)
        super().delete_node(label, node_gid)
        self._log("delete_node", {"label": label, "node_gid": node_gid}, touched)

    def restore_rel(self, label, node_gid, direction, rtype, other_label, other_gid, props):
        super().restore_rel(label, node_gid, direction, rtype, other_label, other_gid, props)
        self._log("restore_rel", {"label": label, "node_gid": node_gid, "direction": direction, "rtype": rtype,
                                  "other_label": other_label, "other_gid": other_gid, "props": props}, [])

    def restore_edge(self, rtype, edge_gid, src, dst, props):
        touched = self._edge(rtype, edge_gid)
        super().restore_edge(rtype, edge_gid, src, dst, props)
        self._log("restore_edge", {"rtype": rtype, "edge_gid": edge_gid, "src": src, "dst": dst, "props": props},
                  touched)

    def delete_edge(self, rtype, edge_gid):
        touched = self._edge(rtype, edge_gid)
        super().delete_edge(rtype, edge_gid)
        self._log("delete_edge", {"rtype": rtype, "edge_gid": edge_gid}, touched)


def _copy_nodes(store, dst: str, label: str, props_list) -> None:
    for p in props_list:
        if p:
            store.query(dst, f"CREATE (n:{label}) SET n = $p", {"p": p})


def copy_activity(store, src: str, dst: str) -> int:
    """Append-only copy of Runs/Steps/Observations that dst doesn't have (branch -> main on merge)."""
    have = {r[0] for r in store.ro_query(dst, "MATCH (r:Run) RETURN r.gid")}
    runs = [g for (g,) in store.ro_query(src, "MATCH (r:Run) RETURN r.gid") if g not in have]
    if not runs:
        return 0
    p = {"g": runs}
    _copy_nodes(store, dst, "Run", [r[0] for r in store.ro_query(
        src, "MATCH (r:Run) WHERE r.gid IN $g RETURN properties(r)", p)])
    _copy_nodes(store, dst, "Step", [r[0] for r in store.ro_query(
        src, "MATCH (s:Step)-[:IN_RUN]->(r:Run) WHERE r.gid IN $g RETURN properties(s)", p)])
    _copy_nodes(store, dst, "Observation", [r[0] for r in store.ro_query(
        src, "MATCH (o:Observation)-[:DURING]->(:Step)-[:IN_RUN]->(r:Run) WHERE r.gid IN $g RETURN properties(o)", p)])
    links = [
        ("MATCH (s:Step)-[:IN_RUN]->(r:Run) WHERE r.gid IN $g RETURN s.gid, r.gid", "Step", "IN_RUN", "Run"),
        ("MATCH (a:Step)-[:NEXT]->(b:Step)-[:IN_RUN]->(r:Run) WHERE r.gid IN $g RETURN a.gid, b.gid", "Step", "NEXT",
         "Step"),
        ("MATCH (s:Step)-[:FROM]->(t:State), (s)-[:IN_RUN]->(r:Run) WHERE r.gid IN $g RETURN s.gid, t.gid",
         "Step", "FROM", "State"),
        ("MATCH (s:Step)-[:TO]->(t:State), (s)-[:IN_RUN]->(r:Run) WHERE r.gid IN $g RETURN s.gid, t.gid",
         "Step", "TO", "State"),
        ("MATCH (o:Observation)-[:DURING]->(s:Step)-[:IN_RUN]->(r:Run) WHERE r.gid IN $g RETURN o.gid, s.gid",
         "Observation", "DURING", "Step"),
        ("MATCH (r:Run)-[:EXECUTED]->(w:Workflow) WHERE r.gid IN $g RETURN r.gid, w.gid", "Run", "EXECUTED",
         "Workflow"),
    ]
    for cypher, la, rt, lb in links:
        for a, b in store.ro_query(src, cypher, p):
            store.query(dst, f"MATCH (a:{la} {{gid:$a}}), (b:{lb} {{gid:$b}}) CREATE (a)-[:{rt}]->(b)",
                        {"a": a, "b": b})
    for og, label, tg in store.ro_query(src, """MATCH (o:Observation)-[:DURING]->(:Step)-[:IN_RUN]->(r:Run), (o)-[:OF]->(t)
                                                WHERE r.gid IN $g RETURN o.gid, labels(t)[0], t.gid""", p):
        store.query(dst, f"MATCH (o:Observation {{gid:$o}}), (t:{label} {{gid:$t}}) CREATE (o)-[:OF]->(t)",
                    {"o": og, "t": tg})
    return len(runs)
