from __future__ import annotations

import json

from ..capture.model import StateObs
from ..capture.segment import jaccard
from ..config import Settings
from ..db import GraphStore
from ..ids import gid, now_ms

_COST = """WITH r
SET r.reliability = toFloat(r.success_count + 1) / (r.seen_count + 2)
WITH r
SET r.cost = 1.0 + $lam * (0.0 - log(r.reliability))"""

_PAGE = """MERGE (p:Page {gid:$pg})
SET p.url_template = $tpl, p.title = $title, p.first_seen = coalesce(p.first_seen, $now), p.last_seen = $now,
    p.seen_count = coalesce(p.seen_count, 0) + 1,
    p.example_urls = CASE WHEN p.example_urls IS NULL THEN [$url]
                          WHEN $url IN p.example_urls OR size(p.example_urls) >= 5 THEN p.example_urls
                          ELSE p.example_urls + [$url] END"""

_STATE = """MATCH (p:Page {gid:$pg})
MERGE (s:State {gid:$sg})
SET s.first_seen = coalesce(s.first_seen, $now), s.last_seen = $now, s.seen_count = coalesce(s.seen_count, 0) + 1,
    s.tokens = coalesce(s.tokens, $tokens), s.skeleton_hash = coalesce(s.skeleton_hash, $sk),
    s.blocking_sig = coalesce(s.blocking_sig, $bsig), s.aliases = coalesce(s.aliases, []),
    s.untried_count = coalesce(s.untried_count, 0),
    s.screenshot_ref = coalesce(s.screenshot_ref, $shot), s.snapshot_ref = coalesce(s.snapshot_ref, $snap)
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
    x.input_kind = e.input_kind, x.href_template = e.href_template, x.in_form = e.in_form,
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


class MapWriter:
    def __init__(self, store: GraphStore, key: str, settings: Settings, now=now_ms):
        self.store, self.key, self.s, self.now = store, key, settings, now

    def q(self, cypher: str, **params):
        return self.store.query(self.key, cypher, params)

    def ro(self, cypher: str, **params):
        return self.store.ro_query(self.key, cypher, params)

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

    def write_state(self, obs: StateObs, screenshot_ref: str | None = None, snapshot_ref: str | None = None) -> str:
        now = self.now()
        canon = self.resolve_state(obs)
        new = canon is None
        sg = canon or obs.state_gid
        self.q(_PAGE, pg=obs.page_gid, tpl=obs.url_template, url=obs.url, title=obs.title, now=now)
        self.q(_STATE, pg=obs.page_gid, sg=sg, tokens=obs.tokens, sk=obs.skeleton_hash, bsig=obs.blocking_sig,
               shot=screenshot_ref, snap=snapshot_ref, now=now)
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

    def write_action(self, src: str, dst: str, el: dict, action_type: str = "click", value_var: str | None = None,
                     pre_cstates=(), requires_inputs=(), success: bool = True) -> str:
        g = gid("ac", src, el["gid"], action_type, value_var or "", dst)
        label = f"{action_type} '{(el.get('name_sample') or el.get('name_norm') or '')[:40]}'"
        self.q(_ACTION, src=src, dst=dst, g=g, el=el["gid"], type=action_type, var=value_var, label=label,
               risk=el.get("risk", "submit_safe"), rc=int(el.get("risk_cost", 1)), pre=list(pre_cstates),
               inputs=list(requires_inputs), succ=1 if success else 0, lam=self.s.cost_lambda, now=self.now())
        return g

    def record_attempt(self, action_gid: str, success: bool) -> None:
        self.q(_ATTEMPT, g=action_gid, succ=1 if success else 0, lam=self.s.cost_lambda, now=self.now())

    def write_transition(self, comp_gid: str, src_cs: str, dst_cs: str, el: dict, action_type: str = "click",
                         src_name: str = "absent", dst_name: str = "open") -> str:
        g = gid("tr", src_cs, el["gid"], action_type, dst_cs)
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

    def promote_locator(self, el_gid: str, locator: str) -> None:
        self.q("MATCH (x:Element {gid:$g}) SET x.locators = [$l] + [y IN x.locators WHERE y <> $l], "
               "x.heal_count = x.heal_count + 1", g=el_gid, l=locator)
