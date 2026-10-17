"""Re-hash migration (spec §3.7): rebuild the map layer of a device graph from the stored raw snapshots with the
current identity rules, carrying workflows, intents, activity and demos over to the new gids."""
from __future__ import annotations

import gzip
import json

from .capture.model import RawSnapshot
from .capture.segment import build_state
from .graph.map_writer import MapWriter
from .ids import gid, now_ms
from .schema import ensure_device_schema
from .versions import ALGO_VERSION

_CARRY = ["Workflow", "WStep", "Intent", "Feature", "Run", "Step", "Observation", "DemoRequest", "Storyboard", "Shot",
          "Render", "Site"]


def _props(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "emb"}


def _remap_action_lists(props: dict, el_map: dict[str, str]) -> dict:
    """Element gids also sit inside lists on an action: the inputs it needs and the steps learned from recoveries."""
    p = dict(props)
    if p.get("requires_inputs"):
        p["requires_inputs"] = [el_map.get(g, g) for g in p["requires_inputs"]]
    if p.get("pre_actions"):
        p["pre_actions"] = [f"{k}|{el_map.get(g, g)}" for k, _, g in (t.partition("|") for t in p["pre_actions"])]
    return p


def rehash(store, key: str, blobs, settings, domain: str) -> dict:
    """Rebuild the map layer from stored raw snapshots with the current identity rules. The old graph is kept as a
    tag (`backup` in the result)."""
    tag = f"{key}@tag:pre-algo-{now_ms()}"
    store.copy(key, tag)
    tmp = f"{key}@mig"
    store.delete(tmp)
    ensure_device_schema(store, tmp, settings.embed_dim)
    w = MapWriter(store, tmp, settings)
    state_map: dict[str, str] = {}
    el_map: dict[str, str] = {}
    def replay(snap: str, old_elements) -> str:
        raw = RawSnapshot.from_dict(json.loads(gzip.decompress(blobs.read(snap))))
        obs = build_state(raw, domain, settings.template_min_items)
        new_sg = w.write_state(obs, snapshot_ref=snap)
        by_css = {e.css: e.gid for e in obs.elements()}
        by_item = {e.item_locator: e.gid for e in obs.elements() if e.is_template}
        for eg, fp, item in old_elements:
            css = json.loads(fp or "{}").get("css")
            new = by_item.get(item) if item else by_css.get(css)
            if new:
                el_map[eg] = new
        return new_sg

    for sg, snap in store.ro_query(key, "MATCH (s:State) WHERE s.snapshot_ref IS NOT NULL RETURN s.gid, s.snapshot_ref"):
        state_map[sg] = replay(snap, store.ro_query(
            key, """MATCH (s:State {gid:$g})-[:COMPOSED_OF]->(:ComponentState)-[:CONTAINS]->(e:Element)
                    RETURN e.gid, e.fingerprint, e.item_locator""", {"g": sg}))
    # Component states first seen after their state (a menu opened): their elements are only in their own snapshot.
    for (snap,) in store.ro_query(key, "MATCH (c:ComponentState) WHERE c.snapshot_ref IS NOT NULL "
                                       "RETURN DISTINCT c.snapshot_ref"):
        replay(snap, store.ro_query(
            key, """MATCH (c:ComponentState {snapshot_ref:$r})-[:CONTAINS]->(e:Element)
                    RETURN e.gid, e.fingerprint, e.item_locator""", {"r": snap}))
        for (cg,) in store.ro_query(key, "MATCH (c:ComponentState {snapshot_ref:$r}) RETURN c.gid", {"r": snap}):
            store.query(tmp, "MATCH (c:ComponentState {gid:$g}) SET c.snapshot_ref = $r", {"g": cg, "r": snap})
    for sg, label in store.ro_query(key, "MATCH (s:State) RETURN s.gid, labels(s)"):
        if sg in state_map:
            for extra in ("Entry", "Blocking"):
                if extra in label:
                    store.query(tmp, f"MATCH (s:State {{gid:$g}}) SET s:{extra}", {"g": state_map[sg]})
    action_map: dict[str, str] = {}
    dropped_actions = 0
    for g, a, b, props in store.ro_query(key, "MATCH (a:State)-[r:ACTION]->(b:State) RETURN r.gid, a.gid, b.gid, properties(r)"):
        if not (a in state_map and b in state_map and props.get("el_gid") in el_map):
            dropped_actions += 1
        if a in state_map and b in state_map and props.get("el_gid") in el_map:
            ng = gid("ac", state_map[a], el_map[props["el_gid"]], props.get("action_type"), props.get("value_var") or "",
                     state_map[b])
            p = {**_remap_action_lists(props, el_map), "gid": ng, "el_gid": el_map[props["el_gid"]]}
            store.query(tmp, "MATCH (a:State {gid:$a}), (b:State {gid:$b}) MERGE (a)-[r:ACTION {gid:$g}]->(b) SET r += $p",
                        {"a": state_map[a], "b": state_map[b], "g": ng, "p": p})
            action_map[g] = ng
    dropped_transitions = 0
    for g, a, b, props in store.ro_query(key, "MATCH (a:ComponentState)-[r:TRANSITION]->(b:ComponentState) "
                                              "RETURN r.gid, a.gid, b.gid, properties(r)"):
        if props.get("el_gid") in el_map:
            store.query(tmp, """MERGE (a:ComponentState {gid:$a}) MERGE (b:ComponentState {gid:$b})
                                MERGE (a)-[r:TRANSITION {gid:$g}]->(b) SET r += $p""",
                        {"a": a, "b": b, "g": g, "p": {**props, "el_gid": el_map[props["el_gid"]]}})
            action_map[g] = g
        else:
            dropped_transitions += 1
    remap = {**state_map, **el_map, **action_map}
    for label in _CARRY:
        for (props,) in store.ro_query(key, f"MATCH (n:{label}) RETURN properties(n)"):
            p = _props(props)
            for f in ("action_gid", "transition_gid", "medoid_gid"):
                if p.get(f) in remap:
                    p[f] = remap[p[f]]
            store.query(tmp, f"CREATE (n:{label}) SET n = $p", {"p": p})
            emb = props.get("emb")
            if emb is not None:
                store.query(tmp, f"MATCH (n:{label} {{gid:$g}}) SET n.emb = vecf32($v)", {"g": p["gid"], "v": list(emb)})
    rels = store.ro_query(key, """MATCH (a)-[r]->(b)
        WHERE (labels(a)[0] IN $L OR labels(b)[0] IN $L) AND NOT type(r) IN ['ACTION', 'TRANSITION']
        RETURN labels(a)[0], a.gid, type(r), properties(r), labels(b)[0], b.gid""", {"L": _CARRY})
    for la, ga, t, props, lb, gb in rels:
        ga, gb = remap.get(ga, ga), remap.get(gb, gb)
        store.query(tmp, f"MATCH (a:{la} {{gid:$a}}), (b:{lb} {{gid:$b}}) CREATE (a)-[r:{t}]->(b) SET r = $p",
                    {"a": ga, "b": gb, "p": props})
    store.query(tmp, "MATCH (s:Site) SET s.algo_version = $v", {"v": ALGO_VERSION})
    store.delete(key)
    store.copy(tmp, key)
    store.delete(tmp)
    return {"states": len(set(state_map.values())), "elements": len(el_map), "actions": len(action_map),
            "dropped_actions": dropped_actions, "dropped_transitions": dropped_transitions, "backup": tag}
