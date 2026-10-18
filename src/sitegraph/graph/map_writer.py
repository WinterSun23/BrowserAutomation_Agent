from __future__ import annotations

import json

from ..capture.model import StateObs
from ..capture.segment import jaccard
from ..config import Settings
from ..db import GraphStore
from ..ids import gid, now_ms
from ..normalize import normalize_text
from ..versions import ALGO_VERSION, SCHEMA_VERSION

_COST = """WITH r
SET r.reliability = toFloat(r.success_count + 1) / (r.seen_count + 2)
WITH r
SET r.cost = 1.0 + $lam * (0.0 - log(r.reliability))"""

_PAGE = """MERGE (p:Page {gid:$pg})
SET p.url_template = $tpl, p.title = $title, p.first_seen = coalesce(p.first_seen, $now), p.last_seen = $now,
    p.seen_count = coalesce(p.seen_count, 0) + 1,
    p.example_urls = CASE WHEN p.example_urls IS NULL THEN [$url]
                          WHEN $url IN p.example_urls OR size(p.example_urls) >= 5 THEN p.example_urls
                          ELSE p.example_urls + [$url] END,
    p.example_titles = CASE WHEN p.example_titles IS NULL THEN [$title]
                            WHEN $title IN p.example_titles OR size(p.example_titles) >= 8 THEN p.example_titles
                            ELSE p.example_titles + [$title] END"""

_STATE = """MATCH (p:Page {gid:$pg})
MERGE (s:State {gid:$sg})
SET s.first_seen = coalesce(s.first_seen, $now), s.last_seen = $now, s.seen_count = coalesce(s.seen_count, 0) + 1,
    s.tokens = coalesce(s.tokens, $tokens), s.skeleton_hash = coalesce(s.skeleton_hash, $sk),
    s.blocking_sig = coalesce(s.blocking_sig, $bsig), s.aliases = coalesce(s.aliases, []),
    s.untried_count = coalesce(s.untried_count, 0),
    s.screenshot_ref = coalesce(s.screenshot_ref, $shot), s.snapshot_ref = coalesce(s.snapshot_ref, $snap),
    s.phash = coalesce(s.phash, $phash)
MERGE (p)-[:HAS_STATE]->(s)"""

_COMPONENTS = """UNWIND $comps AS c
MERGE (k:Component {gid:c.gid})
SET k.kind = c.kind, k.name = coalesce(k.name, c.name), k.blocking = c.blocking, k.ephemeral = c.ephemeral,
    k.root_css = c.root_css, k.first_seen = coalesce(k.first_seen, $now), k.last_seen = $now
MERGE (cs:ComponentState {gid:c.cs_gid})
SET cs.name = coalesce(cs.name, c.cs_name), cs.skeleton_hash = c.skeleton_hash, cs.aria_state = c.aria_state,
    cs.is_default = coalesce(cs.is_default, c.is_default), cs.first_seen = coalesce(cs.first_seen, $now),
    cs.last_seen = $now
MERGE (k)-[:HAS_CSTATE]->(cs)"""

_COMPOSE = """UNWIND $cs AS g
MATCH (s:State {gid:$sg}), (cs:ComponentState {gid:g})
MERGE (s)-[:COMPOSED_OF]->(cs)"""

_ELEMENTS = """UNWIND $els AS e
MATCH (cs:ComponentState {gid:e.cs_gid})
MERGE (x:Element {gid:e.gid})
SET x.first_seen = coalesce(x.first_seen, $now), x.last_seen = $now, x.tried = coalesce(x.tried, false),
    x.heal_count = coalesce(x.heal_count, 0), x.role = e.role, x.name_norm = e.name_norm, x.tag = e.tag,
    x.is_template = e.is_template, x.count = e.count, x.risk = e.risk, x.risk_cost = e.risk_cost,
    x.input_kind = e.input_kind, x.href_template = e.href_template, x.in_form = e.in_form, x.inferred = e.inferred,
    x.item_locator = e.item_locator, x.item_key_fields = e.item_key_fields, x.sample_items = e.sample_items,
    x.fingerprint = e.fingerprint, x.locators = CASE WHEN x.locators IS NULL THEN e.locators
        ELSE x.locators + [l IN e.locators WHERE NOT l IN x.locators] END
SET x.volatile = coalesce(x.volatile, false) OR (x.name_sample IS NOT NULL AND x.name_sample <> e.name_sample)
SET x.name_sample = e.name_sample, x.sample_ts = $now
MERGE (cs)-[:CONTAINS]->(x)"""

_ACTION = """MATCH (a:State {gid:$src}), (b:State {gid:$dst})
MERGE (a)-[r:ACTION {gid:$g}]->(b)
SET r.first_seen = coalesce(r.first_seen, $now), r.el_gid = $el, r.action_type = $type, r.value_var = $var,
    r.label = $label, r.risk = $risk, r.risk_cost = $rc, r.pre_cstates = coalesce(r.pre_cstates, $pre),
    r.requires_inputs = coalesce(r.requires_inputs, $inputs),
    r.seen_count = coalesce(r.seen_count, 0) + 1, r.success_count = coalesce(r.success_count, 0) + $succ,
    r.last_seen = $now
""" + _COST + "\nRETURN r.gid"

_ATTEMPT = """MATCH ()-[r:ACTION {gid:$g}]->()
SET r.seen_count = r.seen_count + 1, r.success_count = r.success_count + $succ, r.last_seen = $now
""" + _COST

_TRANSITION = """MERGE (k:Component {gid:$comp})
MERGE (a:ComponentState {gid:$src})
SET a.name = coalesce(a.name, $src_name), a.first_seen = coalesce(a.first_seen, $now)
MERGE (b:ComponentState {gid:$dst})
SET b.name = coalesce(b.name, $dst_name), b.first_seen = coalesce(b.first_seen, $now)
MERGE (k)-[:HAS_CSTATE]->(a)
MERGE (k)-[:HAS_CSTATE]->(b)
MERGE (a)-[t:TRANSITION {gid:$g}]->(b)
SET t.first_seen = coalesce(t.first_seen, $now), t.el_gid = $el, t.action_type = $type, t.label = $label,
    t.risk_cost = $rc, t.seen_count = coalesce(t.seen_count, 0) + 1,
    t.success_count = coalesce(t.success_count, 0) + 1, t.last_seen = $now
WITH t
SET t.reliability = toFloat(t.success_count + 1) / (t.seen_count + 2)
RETURN t.gid"""

_FRONTIER = """MATCH (s:State {gid:$g})
OPTIONAL MATCH (s)-[:COMPOSED_OF]->(:ComponentState)-[:CONTAINS]->(x:Element)
  WHERE x.tried = false AND x.risk_cost <= $mrc AND x.risk <> 'input'
WITH s, count(DISTINCT x) AS n
SET s.untried_count = n
RETURN n"""


_ENDPOINT = {"ACTION": "State", "BROKEN_ACTION": "State", "TRANSITION": "ComponentState",
             "BROKEN_TRANSITION": "ComponentState"}
_BROKEN = {"ACTION": "BROKEN_ACTION", "TRANSITION": "BROKEN_TRANSITION"}
_REPOINT_IN = ["FROM", "TO", "SHOWS", "FULFILLED_BY"]   # (x)-[r]->(State) relationships that follow a merged state


def _clean(p: dict | None) -> dict:
    return {k: v for k, v in (p or {}).items() if v is not None and k != "emb"}


class MapWriter:
    def __init__(self, store: GraphStore, key: str, settings: Settings, now=now_ms):
        self.store, self.key, self.s, self.now = store, key, settings, now

    def q(self, cypher: str, **params):
        return self.store.query(self.key, cypher, params)

    def ro(self, cypher: str, **params):
        return self.store.ro_query(self.key, cypher, params)

    def ensure_site(self, domain: str, device: str, device_profile: str, embed_model: str, embed_dim: int) -> dict:
        """The Site node records which schema, identity rules and embedding model built this graph."""
        rows = self.ro("MATCH (s:Site) RETURN s.embed_model, s.embed_dim, s.algo_version, s.schema_version LIMIT 1")
        if rows and rows[0][0] and (rows[0][0] != embed_model or rows[0][1] != embed_dim):
            raise ValueError(f"graph {self.key} was built with {rows[0][0]}/{rows[0][1]}, not {embed_model}/{embed_dim}")
        self.q("""MERGE (s:Site {gid:$g})
                  SET s.domain = $d, s.device = $dev, s.device_profile = $p, s.embed_model = $m, s.embed_dim = $dim,
                      s.schema_version = $sv, s.algo_version = coalesce(s.algo_version, $av),
                      s.first_seen = coalesce(s.first_seen, $now)""",
               g=gid("site", domain, device), d=domain, dev=device, p=device_profile, m=embed_model, dim=embed_dim,
               sv=SCHEMA_VERSION, av=ALGO_VERSION, now=self.now())
        r = self.ro("MATCH (s:Site) RETURN s.algo_version, s.schema_version LIMIT 1")[0]
        return {"algo_version": r[0], "schema_version": r[1]}

    def resolve_state(self, obs: StateObs) -> str | None:
        if self.ro("MATCH (s:State {gid:$g}) RETURN s.gid", g=obs.state_gid):
            return obs.state_gid
        rows = self.ro("MATCH (:Page {gid:$pg})-[:HAS_STATE]->(s:State) "
                       "RETURN s.gid, s.aliases, s.tokens, s.blocking_sig", pg=obs.page_gid)
        for g, aliases, _, _ in rows:
            if obs.state_gid in (aliases or []):
                return g
        mine = set(obs.tokens)
        best, best_sim = None, 0.0
        for g, _, tokens, bsig in rows:
            if (bsig or "") != obs.blocking_sig:
                continue
            sim = jaccard(mine, set(tokens or []))
            if sim >= self.s.near_dup_threshold and sim > best_sim:
                best, best_sim = g, sim
        return best

    def write_state(self, obs: StateObs, screenshot_ref: str | None = None, snapshot_ref: str | None = None,
                    phash: str | None = None) -> str:
        now = self.now()
        canon = self.resolve_state(obs)
        new = canon is None
        sg = canon or obs.state_gid
        self.q(_PAGE, pg=obs.page_gid, tpl=obs.url_template, url=obs.url, title=obs.title, now=now)
        self.q(_STATE, pg=obs.page_gid, sg=sg, tokens=obs.tokens, sk=obs.skeleton_hash, bsig=obs.blocking_sig,
               shot=screenshot_ref, snap=snapshot_ref, phash=phash, now=now)
        if canon and canon != obs.state_gid:
            self.q("MATCH (s:State {gid:$sg}) WHERE NOT $a IN s.aliases SET s.aliases = s.aliases + [$a]",
                   sg=sg, a=obs.state_gid)
        if obs.blocking_sig:
            self.q("MATCH (s:State {gid:$sg}) SET s:Blocking", sg=sg)
        comps = [dict(gid=c.gid, kind=c.kind, name=c.name, blocking=c.blocking, ephemeral=c.ephemeral,
                      root_css=c.root_css, cs_gid=c.cs_gid, cs_name=c.cs_name, skeleton_hash=c.skeleton_hash,
                      aria_state=json.dumps(c.aria_state, sort_keys=True), is_default=new or not c.ephemeral)
                 for c in obs.components]
        if comps:
            self.q(_COMPONENTS, comps=comps, now=now)
            compose = [c.cs_gid for c in obs.components if new or not c.ephemeral]
            self.q(_COMPOSE, cs=compose, sg=sg)
        els = [e.props() for e in obs.elements()]
        if els:
            self.q(_ELEMENTS, els=els, now=now)
        return sg

    def cstates_without_shot(self, gids: list[str]) -> set[str]:
        if not gids:
            return set()
        rows = self.ro("MATCH (c:ComponentState) WHERE c.gid IN $g AND c.screenshot_ref IS NULL RETURN c.gid", g=gids)
        return {r[0] for r in rows}

    def set_cstate_shot(self, cs_gid: str, ref: str) -> None:
        self.q("MATCH (c:ComponentState {gid:$g}) WHERE c.screenshot_ref IS NULL SET c.screenshot_ref = $r",
               g=cs_gid, r=ref)

    def set_cstate_snapshot(self, cs_gid: str, ref: str) -> None:
        """Raw snapshot in which a component state was first seen, so a re-hash can rebuild it later."""
        self.q("MATCH (c:ComponentState {gid:$g}) WHERE c.snapshot_ref IS NULL SET c.snapshot_ref = $r",
               g=cs_gid, r=ref)

    def write_action(self, src: str, dst: str, el: dict, action_type: str = "click", value_var: str | None = None,
                     pre_cstates=(), requires_inputs=(), success: bool = True) -> str:
        g = self.action_gid(src, el["gid"], action_type, value_var, dst)
        label = f"{action_type} '{(el.get('name_sample') or el.get('name_norm') or '')[:40]}'"
        self.q(_ACTION, src=src, dst=dst, g=g, el=el["gid"], type=action_type, var=value_var, label=label,
               risk=el.get("risk", "submit_safe"), rc=int(el.get("risk_cost", 1)), pre=list(pre_cstates),
               inputs=list(requires_inputs), succ=1 if success else 0, lam=self.s.cost_lambda, now=self.now())
        return g

    def record_attempt(self, action_gid: str, success: bool) -> None:
        self.q(_ATTEMPT, g=action_gid, succ=1 if success else 0, lam=self.s.cost_lambda, now=self.now())

    def add_pre_actions(self, action_gid: str, actions: list[str]) -> None:
        """Learned steps that must come before this action when their element is on the page (e.g. closing a banner,
        choosing a size); tokens are 'click|<el_gid>' or 'hover|<el_gid>'."""
        if actions:
            self.q("MATCH ()-[r:ACTION {gid:$g}]->() "
                   "SET r.pre_actions = coalesce(r.pre_actions, []) + [a IN $acts WHERE NOT a IN coalesce(r.pre_actions, [])]",
                   g=action_gid, acts=list(dict.fromkeys(actions)))

    def pre_actions(self, action_gid: str) -> list[str]:
        rows = self.ro("MATCH ()-[r:ACTION {gid:$g}]->() RETURN r.pre_actions", g=action_gid)
        return list(rows[0][0] or []) if rows else []

    def write_transition(self, comp_gid: str, src_cs: str, dst_cs: str, el: dict, action_type: str = "click",
                         src_name: str = "absent", dst_name: str = "open") -> str:
        g = self.transition_gid(src_cs, el["gid"], action_type, dst_cs)
        label = f"{action_type} '{(el.get('name_sample') or '')[:40]}'"
        self.q(_TRANSITION, comp=comp_gid, src=src_cs, dst=dst_cs, src_name=src_name, dst_name=dst_name, g=g,
               el=el["gid"], type=action_type, label=label, rc=int(el.get("risk_cost", 1)), now=self.now())
        return g

    def mark_tried(self, el_gid: str) -> None:
        self.q("MATCH (x:Element {gid:$g}) SET x.tried = true", g=el_gid)

    def set_entry(self, state_gid: str) -> None:
        self.q("MATCH (s:State {gid:$g}) SET s:Entry", g=state_gid)

    def refresh_frontier(self, state_gid: str, max_risk_cost: int) -> int:
        rows = self.q(_FRONTIER, g=state_gid, mrc=max_risk_cost)
        n = rows[0][0] if rows else 0
        self.q("MATCH (s:State {gid:$g}) " + ("SET s:Frontier" if n else "REMOVE s:Frontier"), g=state_gid)
        return n

    def bump_locator(self, el_gid: str, locator: str, healed: bool = False) -> None:
        """Count a locator that found the element (`Element.locator_hits`, JSON) and order the locators by hits, so the
        ones that keep working are tried first. `healed`: it was found by a fallback, not the first locator."""
        rows = self.ro("MATCH (x:Element {gid:$g}) RETURN x.locators, x.locator_hits", g=el_gid)
        if not rows:
            return
        locs, hits = rows[0][0] or [], json.loads(rows[0][1] or "{}")
        hits[locator] = hits.get(locator, 0) + 1
        if locator not in locs:
            locs = [locator] + locs
        locs = sorted(locs, key=lambda l: -hits.get(l, 0))
        self.q("MATCH (x:Element {gid:$g}) SET x.locators = $l, x.locator_hits = $h, "
               "x.heal_count = coalesce(x.heal_count, 0) + $healed",
               g=el_gid, l=locs, h=json.dumps(hits, sort_keys=True), healed=1 if healed else 0)

    # ---------- generic, recordable primitives (used by merge, revert, repair) ----------
    @staticmethod
    def action_gid(src, el_gid, action_type, value_var, dst) -> str:
        return gid("ac", src, el_gid, action_type, value_var or "", dst)

    @staticmethod
    def transition_gid(src_cs, el_gid, action_type, dst_cs) -> str:
        return gid("tr", src_cs, el_gid, action_type, dst_cs)

    def node_props(self, label: str, gids) -> dict:
        gids = list(gids)
        if not gids:
            return {}
        found = {g: _clean(p) for g, p in self.ro(f"MATCH (n:{label}) WHERE n.gid IN $g RETURN n.gid, properties(n)",
                                                  g=gids)}
        return {g: found.get(g) for g in gids}

    def edge_props(self, rtype: str, edge_gid: str) -> dict | None:
        rows = self.ro(f"MATCH (a)-[r:{rtype} {{gid:$g}}]->(b) RETURN properties(r), a.gid, b.gid", g=edge_gid)
        return {"props": _clean(rows[0][0]), "src": rows[0][1], "dst": rows[0][2]} if rows else None

    def restore_node(self, label: str, node_gid: str, props: dict) -> None:
        self.q(f"MERGE (n:{label} {{gid:$g}}) WITH n, n.emb AS e SET n = $p SET n.emb = e",
               g=node_gid, p={**_clean(props), "gid": node_gid})

    def delete_node(self, label: str, node_gid: str) -> None:
        self.q(f"MATCH (n:{label} {{gid:$g}}) DETACH DELETE n", g=node_gid)

    def restore_rel(self, label: str, node_gid: str, direction: str, rtype: str, other_label: str, other_gid: str,
                    props: dict) -> None:
        pattern = f"(n)-[r:{rtype}]->(m)" if direction == "out" else f"(n)<-[r:{rtype}]-(m)"
        self.q(f"MATCH (n:{label} {{gid:$a}}), (m:{other_label} {{gid:$b}}) MERGE {pattern} SET r += $p",
               a=node_gid, b=other_gid, p=_clean(props))

    def restore_edge(self, rtype: str, edge_gid: str, src: str, dst: str, props: dict) -> None:
        lab = _ENDPOINT[rtype]
        self.q(f"MATCH (a:{lab} {{gid:$s}}), (b:{lab} {{gid:$d}}) MERGE (a)-[r:{rtype} {{gid:$g}}]->(b) SET r = $p",
               s=src, d=dst, g=edge_gid, p={**_clean(props), "gid": edge_gid})

    def delete_edge(self, rtype: str, edge_gid: str) -> None:
        self.q(f"MATCH ()-[r:{rtype} {{gid:$g}}]->() DELETE r", g=edge_gid)

    def set_props(self, label: str, node_gid: str, props: dict) -> None:
        self.q(f"MATCH (n:{label} {{gid:$g}}) SET n += $p", g=node_gid, p=_clean(props))

    def quarantine(self, rtype: str, edge_gid: str, reason: str, superseded_by: str | None = None) -> bool:
        e = self.edge_props(rtype, edge_gid)
        if e is None:
            return False
        self.delete_edge(rtype, edge_gid)
        props = {**e["props"], "broken_since": self.now(), "reason": reason}
        if superseded_by:
            props["superseded_by"] = superseded_by
        self.restore_edge(_BROKEN[rtype], edge_gid, e["src"], e["dst"], props)
        return True

    def unquarantine(self, rtype: str, edge_gid: str) -> bool:
        e = self.edge_props(_BROKEN[rtype], edge_gid)
        if e is None:
            return False
        self.delete_edge(_BROKEN[rtype], edge_gid)
        props = {k: v for k, v in e["props"].items() if k not in ("broken_since", "reason", "superseded_by")}
        self.restore_edge(rtype, edge_gid, e["src"], e["dst"], props)
        return True

    # ---------- curator edits (the Curator review agent's merge / rename) ----------
    def action_edges_of(self, state_gid: str) -> list[dict]:
        """Every ACTION edge touching a state: {gid, src, dst, props}."""
        out = []
        for direction, q in (
                ("out", "MATCH (b:State {gid:$m})-[r:ACTION]->(o:State) RETURN r.gid, properties(r), b.gid, o.gid"),
                ("in", "MATCH (o:State)-[r:ACTION]->(b:State {gid:$m}) RETURN r.gid, properties(r), o.gid, b.gid")):
            for g, props, src, dst in self.ro(q, m=state_gid):
                if direction == "in" and src == dst:
                    continue                                        # a self-loop was already listed as "out"
                out.append({"gid": g, "src": src, "dst": dst, "props": props})
        return out

    def merge_target_gid(self, edge: dict, keep: str, merge: str) -> str | None:
        """The gid an edge of `merge` gets once re-created on `keep` (None when it would become a self-loop)."""
        src = keep if edge["src"] == merge else edge["src"]
        dst = keep if edge["dst"] == merge else edge["dst"]
        if src == dst:
            return None
        p = edge["props"]
        return self.action_gid(src, p.get("el_gid"), p.get("action_type"), p.get("value_var"), dst)

    def _move_action_edges(self, keep: str, merge: str) -> int:
        moved, lam = 0, self.s.cost_lambda
        for edge in self.action_edges_of(merge):
            new_gid = self.merge_target_gid(edge, keep, merge)
            props = edge["props"]
            if new_gid is not None:
                src = keep if edge["src"] == merge else edge["src"]
                dst = keep if edge["dst"] == merge else edge["dst"]
                same = self.ro("MATCH ()-[x:ACTION {gid:$g}]->() RETURN properties(x)", g=new_gid)
                if same:
                    o = same[0][0]
                    self.q("""MATCH ()-[x:ACTION {gid:$g}]->()
                              SET x.seen_count = $seen, x.success_count = $succ, x.first_seen = $fs, x.last_seen = $ls
                              WITH x SET x.reliability = toFloat(x.success_count + 1) / (x.seen_count + 2)
                              WITH x SET x.cost = 1.0 + $lam * (0.0 - log(x.reliability))""",
                           g=new_gid, lam=lam, seen=o.get("seen_count", 0) + props.get("seen_count", 0),
                           succ=o.get("success_count", 0) + props.get("success_count", 0),
                           fs=min(o.get("first_seen", 0), props.get("first_seen", 0)),
                           ls=max(o.get("last_seen", 0), props.get("last_seen", 0)))
                else:
                    self.q("MATCH (a:State {gid:$s}), (b:State {gid:$d}) CREATE (a)-[x:ACTION]->(b) "
                           "SET x += $p, x.gid = $g", s=src, d=dst, p=props, g=new_gid)
                moved += 1
            self.q("MATCH ()-[x:ACTION {gid:$g}]->() DELETE x", g=edge["gid"])
        return moved

    def merge_states(self, keep: str, merge: str) -> dict:
        """Fold state `merge` into `keep` (same page, same modal state): actions move with summed counters, references
        are re-pointed, aliases / tokens / counts / the Entry label are folded in, then `merge` is deleted."""
        if keep == merge:
            raise ValueError("keep and merge must be different states")
        rows = self.ro("""MATCH (p1:Page)-[:HAS_STATE]->(a:State {gid:$k}), (p2:Page)-[:HAS_STATE]->(b:State {gid:$m})
                          RETURN p1.gid, p2.gid, properties(a), properties(b), labels(b)""", k=keep, m=merge)
        if not rows:
            raise ValueError("unknown state gid")
        page_a, page_b, pa, pb, b_labels = rows[0]
        if page_a != page_b:
            raise ValueError("states of different pages are different screens and cannot be merged")
        if (pa.get("blocking_sig") or "") != (pb.get("blocking_sig") or ""):
            raise ValueError("one state has a modal dialog open and the other not: they are different screens")
        moved = self._move_action_edges(keep, merge)
        for rel in _REPOINT_IN:
            extra = " SET n.score = coalesce(n.score, r.score)" if rel == "FULFILLED_BY" else ""
            self.q(f"MATCH (x)-[r:{rel}]->(m:State {{gid:$m}}), (k:State {{gid:$k}}) "
                   f"MERGE (x)-[n:{rel}]->(k){extra} DELETE r", m=merge, k=keep)
        aliases = list(dict.fromkeys(list(pa.get("aliases") or []) + [merge] + list(pb.get("aliases") or [])))
        tokens = sorted(set(pa.get("tokens") or []) | set(pb.get("tokens") or []))
        self.q("""MATCH (k:State {gid:$k}) SET k.aliases = $al, k.tokens = $tok, k.seen_count = $sc,
                  k.first_seen = $fs, k.last_seen = $ls""", k=keep, al=aliases, tok=tokens,
               sc=(pa.get("seen_count") or 0) + (pb.get("seen_count") or 0),
               fs=min(pa.get("first_seen") or self.now(), pb.get("first_seen") or self.now()),
               ls=max(pa.get("last_seen") or 0, pb.get("last_seen") or 0))
        if "Entry" in b_labels:
            self.q("MATCH (k:State {gid:$k}) SET k:Entry", k=keep)
        self.q("MATCH (b:State {gid:$m}) DETACH DELETE b", m=merge)
        return {"kept": keep, "merged": merge, "actions_moved": moved, "aliases": len(aliases),
                "merged_state": _clean(pb)}

    def rename_state(self, state_gid: str, summary: str, titles: list, state_vec: list | None = None,
                     intents: list | None = None) -> dict:
        """Replace a state's summary (and its embedding / 'open <title>' intents, computed by the caller so that a replay
        needs no model)."""
        rows = self.ro("MATCH (s:State {gid:$g}) RETURN coalesce(s.summary, '')", g=state_gid)
        if not rows:
            raise ValueError(f"unknown state {state_gid}")
        self.q("MATCH (s:State {gid:$g}) SET s.summary = $sum, s.examples_n = $n", g=state_gid, sum=summary, n=len(titles))
        if state_vec is not None:
            self.q("MATCH (s:State {gid:$g}) SET s.emb = vecf32($v)", g=state_gid, v=state_vec)
        for text, vec in intents or []:
            self.q("""MERGE (i:Intent {gid:$ig})
                      SET i.text = coalesce(i.text, $t), i.source = coalesce(i.source, 'llm'),
                          i.hits = coalesce(i.hits, 0), i.emb = vecf32($v)
                      WITH i MATCH (s:State {gid:$g})
                      MERGE (i)-[f:FULFILLED_BY]->(s) SET f.score = coalesce(f.score, 1.0)""",
                   ig=gid("in", normalize_text(text)), t=text, v=vec, g=state_gid)
        return {"state": state_gid, "old_summary": rows[0][0], "new_summary": summary}
