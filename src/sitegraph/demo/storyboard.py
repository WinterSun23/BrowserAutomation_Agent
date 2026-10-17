from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from ..agents.workflows import _edge, load_workflow_steps
from ..graph.queries import RouteStep, best_routes, entry_state, find_nodes, find_targets
from ..ids import now_ms, ulid
from ..llm import complete_json
from .spec import DemoSpec


class DemoError(Exception):
    pass


@dataclass
class ShotPlan:
    i: int
    kind: str          # navigate | micro | still
    caption: str
    hidden: bool
    step: RouteStep | None = None
    transition: dict | None = None   # {gid, el_gid, comp_gid, state_gid}
    state_gid: str | None = None


@dataclass
class Storyboard:
    gid: str
    request_gid: str
    title: str
    scope_kind: str
    shots: list[ShotPlan]
    workflow_gid: str | None = None
    map_commit: str | None = None      # the version of the map the demo was made against (git commit id)


class StoryboardBuilder:
    def __init__(self, store, key, embedder, llm, max_risk_cost: int = 1):
        self.store, self.key, self.embedder, self.llm, self.mrc = store, key, embedder, llm, max_risk_cost

    def ro(self, cypher: str, **params):
        return self.store.ro_query(self.key, cypher, params)

    def _route(self, src: str, dst: str, hidden: bool) -> list[ShotPlan]:
        routes = best_routes(self.store, self.key, src, dst, self.mrc)
        if not routes:
            raise DemoError("no allowed route to the requested screen")
        return [ShotPlan(0, "navigate", s.label, hidden, step=s) for s in routes[0].steps]

    def _entry(self) -> str:
        e = entry_state(self.store, self.key)
        if not e:
            raise DemoError("site has not been explored yet")
        return e[0]

    def build(self, spec: DemoSpec, request_text: str, map_commit: str | None = None) -> Storyboard:
        vec = self.embedder.embed([spec.target])[0]
        entry = self._entry()
        wf_gid = None
        shots: list[ShotPlan] = []
        if spec.scope_kind == "workflow":
            found = find_nodes(self.store, self.key, "Workflow", vec, 1)
            if not found:
                raise DemoError("no saved workflow matches the request")
            wf_gid = found[0][0]
            steps = load_workflow_steps(self.store, self.key, wf_gid)
            first_src = next((s["src"] for s in steps if s["kind"] == "ACTION"), entry)
            shots += self._route(entry, first_src, hidden=True)
            for st in steps:
                if st["kind"] == "ACTION":
                    shots.append(ShotPlan(0, "navigate", st["label"], False, step=RouteStep(
                        action_gid=st["action_gid"], label=st["label"], el_gid=st["el_gid"], src=st["src"],
                        dst=st["dst"], reliability=st["reliability"], pre_cstates=st["pre_cstates"],
                        requires_inputs=st["requires_inputs"], value_var=st["value_var"],
                        action_type=st["action_type"])))
                else:
                    shots.append(ShotPlan(0, "micro", st["label"], False,
                                          transition={"gid": st["action_gid"], "el_gid": st["el_gid"]}))
        elif spec.scope_kind in ("goal", "state"):
            targets = [t for t in find_targets(self.store, self.key, vec) if t.kind == "State"]
            if not targets:
                raise DemoError("no screen matches the request")
            shots += self._route(entry, targets[0].gid, hidden=spec.scope_kind == "state")
            shots.append(ShotPlan(0, "still", targets[0].text, False, state_gid=targets[0].gid))
        elif spec.scope_kind == "transition":
            tr = self._find_transition(vec)
            shots += self._route(entry, tr["state_gid"], hidden=True)
            shots.append(ShotPlan(0, "micro", tr["label"], False, transition=tr, state_gid=tr["state_gid"]))
            shots.append(ShotPlan(0, "still", tr["label"], False, state_gid=tr["state_gid"]))
        else:  # feature
            found = find_nodes(self.store, self.key, "Feature", vec, 1)
            if not found:
                raise DemoError("no site area matches the request")
            members = [r[0] for r in self.ro("""MATCH (f:Feature {gid:$f}) MATCH (s:State)
                WHERE s.community = f.community_id RETURN s.gid ORDER BY s.pagerank DESC LIMIT 4""", f=found[0][0])]
            cur = entry
            for m in members:
                try:
                    shots += self._route(cur, m, hidden=False)
                    cur = m
                except DemoError:
                    continue
            shots.append(ShotPlan(0, "still", found[0][1], False, state_gid=cur))
        for i, s in enumerate(shots):
            s.i = i
        self._caption(shots, spec)
        return self._persist(spec, request_text, shots, wf_gid, map_commit)

    def _find_transition(self, vec) -> dict:
        for comp_gid, label, _ in find_nodes(self.store, self.key, "Component", vec, 5, "label"):
            rows = self.ro("""MATCH (k:Component {gid:$c})-[:HAS_CSTATE]->(a)-[t:TRANSITION]->(b)
                              RETURN t.gid, t.el_gid, t.label ORDER BY t.success_count DESC LIMIT 1""", c=comp_gid)
            if not rows:
                continue
            tgid, el_gid, tlabel = rows[0]
            states = self.ro("""MATCH (s:State)-[:COMPOSED_OF]->(:ComponentState)-[:CONTAINS]->(:Element {gid:$e})
                                RETURN s.gid ORDER BY s.pagerank DESC LIMIT 1""", e=el_gid)
            if states:
                return {"gid": tgid, "el_gid": el_gid, "comp_gid": comp_gid, "state_gid": states[0][0],
                        "label": tlabel or label}
        raise DemoError("no recorded interaction matches the request")

    def _caption(self, shots: list[ShotPlan], spec: DemoSpec) -> None:
        visible = [s for s in shots if not s.hidden]
        if not visible:
            return
        listing = "\n".join(f"{n + 1}. {s.kind}: {s.caption}" for n, s in enumerate(visible))
        caps = complete_json(self.llm, "You write short on-screen captions for product demo videos.",
                             f"Audience: {spec.audience}\nShots:\n{listing}\n"
                             "Return a JSON list with one caption (max 12 words) per shot, same order.",
                             default=[])
        if isinstance(caps, list) and len(caps) == len(visible):
            for s, c in zip(visible, caps):
                s.caption = str(c)[:120]
        else:
            for s in visible:
                s.caption = s.caption[:1].upper() + s.caption[1:]

    def _persist(self, spec: DemoSpec, request_text: str, shots: list[ShotPlan], wf_gid: str | None,
                 map_commit: str | None = None) -> Storyboard:
        now = now_ms()
        rg, sg = ulid("dr"), ulid("sb")
        v = self.embedder.embed([request_text])[0]
        title = spec.target[:1].upper() + spec.target[1:]
        self.store.query(self.key, """CREATE (r:DemoRequest {gid:$rg, text:$t, emb:vecf32($v), interpretation:$i, ts:$now})
            -[:PRODUCED]->(:Storyboard {gid:$sg, title:$title, scope_kind:$sk, status:'ok', version:1, ts:$now,
                                                                  map_commit:$mc})""",
                         {"rg": rg, "t": request_text, "v": v, "i": json.dumps(asdict(spec)), "now": now, "sg": sg,
                          "title": title, "sk": spec.scope_kind, "mc": map_commit})
        if wf_gid:
            self.store.query(self.key, "MATCH (sb:Storyboard {gid:$sg}), (w:Workflow {gid:$w}) "
                                       "CREATE (sb)-[:BASED_ON]->(w)", {"sg": sg, "w": wf_gid})
        for s in shots:
            shg = ulid("sh")
            show = s.state_gid or (s.step.dst if s.step else None)
            self.store.query(self.key, """MATCH (sb:Storyboard {gid:$sg})
                CREATE (:Shot {gid:$g, i:$i, kind:$k, caption:$c, hidden:$h, action_gid:$ag, transition_gid:$tg})
                       -[:IN_STORYBOARD]->(sb)""",
                             {"sg": sg, "g": shg, "i": s.i, "k": s.kind, "c": s.caption, "h": s.hidden,
                              "ag": s.step.action_gid if s.step else None,
                              "tg": s.transition["gid"] if s.transition else None})
            if show:
                self.store.query(self.key, "MATCH (sh:Shot {gid:$g}), (st:State {gid:$s}) CREATE (sh)-[:SHOWS]->(st)",
                                 {"g": shg, "s": show})
            s.transition = {**s.transition, "shot_gid": shg} if s.transition else None
            setattr(s, "gid", shg)
        return Storyboard(sg, rg, title, spec.scope_kind, shots, wf_gid, map_commit)


def load_storyboard(store, key, sb_gid: str) -> Storyboard:
    """Rebuild a persisted storyboard (for a re-render): its shots, the steps they replay and the map commit."""
    head = store.ro_query(key, """MATCH (r:DemoRequest)-[:PRODUCED]->(sb:Storyboard {gid:$g})
                                  OPTIONAL MATCH (sb)-[:BASED_ON]->(w:Workflow)
                                  RETURN r.gid, sb.title, sb.scope_kind, w.gid, sb.map_commit""", {"g": sb_gid})[0]
    shots = []
    for shg, i, kind, cap, hidden, ag, tg, show in store.ro_query(key, """
            MATCH (sh:Shot)-[:IN_STORYBOARD]->(:Storyboard {gid:$g}) OPTIONAL MATCH (sh)-[:SHOWS]->(s:State)
            RETURN sh.gid, sh.i, sh.kind, sh.caption, sh.hidden, sh.action_gid, sh.transition_gid, s.gid
            ORDER BY sh.i""", {"g": sb_gid}):
        step = transition = None
        if ag:
            e = _edge(store, key, ag)
            if e:
                step = RouteStep(action_gid=ag, label=e["label"], el_gid=e["el_gid"], src=e["src"], dst=e["dst"],
                                 reliability=e["reliability"], pre_cstates=e["pre_cstates"],
                                 requires_inputs=e["requires_inputs"], value_var=e["value_var"],
                                 action_type=e["action_type"])
        if tg:
            e = _edge(store, key, tg)
            if e:
                transition = {"gid": tg, "el_gid": e["el_gid"], "action_type": e["action_type"]}
        plan = ShotPlan(i, kind, cap, hidden, step=step, transition=transition, state_gid=show)
        setattr(plan, "gid", shg)
        shots.append(plan)
    return Storyboard(sb_gid, head[0], head[1], head[2], shots, head[3], head[4])
