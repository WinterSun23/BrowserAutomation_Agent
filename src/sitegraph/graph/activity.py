from __future__ import annotations

import json

from ..db import GraphStore
from ..ids import now_ms, ulid

_OF_LABELS = {"Element", "State", "ComponentState", "Component"}


class ActivityLog:
    def __init__(self, store: GraphStore, key: str, now=now_ms):
        self.store, self.key, self.now = store, key, now
        self._last_step: dict[str, str] = {}

    def q(self, cypher: str, **params):
        return self.store.query(self.key, cypher, params)

    def start_run(self, kind: str, agent_id: str, task_id: str | None = None, request: str | None = None) -> str:
        g = ulid("run")
        self.q("CREATE (:Run {gid:$g, kind:$kind, agent_id:$agent, task_id:$task, request:$req, status:'running', "
               "started:$now})", g=g, kind=kind, agent=agent_id, task=task_id, req=request, now=self.now())
        return g

    def add_step(self, run_gid: str, i: int, action_type: str, action_gid: str | None, from_gid: str | None,
                 to_gid: str | None, ok: bool, healed: bool = False, heal_method: str | None = None,
                 latency_ms: int = 0, error: str | None = None, value_var: str | None = None,
                 screenshot_ref: str | None = None) -> str:
        g = ulid("step")
        self.q("MATCH (r:Run {gid:$run}) CREATE (st:Step {gid:$g, i:$i, action_type:$type, action_gid:$ag, ok:$ok, "
               "healed:$healed, heal_method:$hm, latency_ms:$lat, error:$err, value_var:$var, ts:$now, "
               "screenshot_ref:$shot})-[:IN_RUN]->(r)",
               run=run_gid, g=g, i=i, type=action_type, ag=action_gid, ok=ok, healed=healed, hm=heal_method,
               lat=latency_ms, err=error, var=value_var, now=self.now(), shot=screenshot_ref)
        if from_gid:
            self.q("MATCH (st:Step {gid:$g}), (s:State {gid:$s}) CREATE (st)-[:FROM]->(s)", g=g, s=from_gid)
        if to_gid:
            self.q("MATCH (st:Step {gid:$g}), (s:State {gid:$s}) CREATE (st)-[:TO]->(s)", g=g, s=to_gid)
        prev = self._last_step.get(run_gid)
        if prev:
            self.q("MATCH (a:Step {gid:$a}), (b:Step {gid:$b}) CREATE (a)-[:NEXT]->(b)", a=prev, b=g)
        self._last_step[run_gid] = g
        return g

    def add_observation(self, kind: str, detail: dict, step_gid: str | None = None, of_gid: str | None = None,
                        of_label: str = "Element", before_ref: str | None = None, after_ref: str | None = None) -> str:
        if of_label not in _OF_LABELS:
            raise ValueError(f"unsupported observation target label {of_label}")
        g = ulid("obs")
        self.q("CREATE (:Observation {gid:$g, kind:$kind, detail:$detail, before_ref:$b, after_ref:$a, ts:$now})",
               g=g, kind=kind, detail=json.dumps(detail, sort_keys=True), b=before_ref, a=after_ref, now=self.now())
        if step_gid:
            self.q("MATCH (o:Observation {gid:$g}), (s:Step {gid:$s}) CREATE (o)-[:DURING]->(s)", g=g, s=step_gid)
        if of_gid:
            self.q(f"MATCH (o:Observation {{gid:$g}}), (x:{of_label} {{gid:$x}}) CREATE (o)-[:OF]->(x)", g=g, x=of_gid)
        return g

    def link_workflow(self, run_gid: str, wf_gid: str) -> None:
        self.q("MATCH (r:Run {gid:$r}), (w:Workflow {gid:$w}) MERGE (r)-[:EXECUTED]->(w)", r=run_gid, w=wf_gid)

    def finish_run(self, run_gid: str, status: str, video_ref: str | None = None, trace_ref: str | None = None) -> None:
        self.q("MATCH (r:Run {gid:$r}) SET r.status = $st, r.ended = $now, r.video_ref = $v, r.trace_ref = $t",
               r=run_gid, st=status, now=self.now(), v=video_ref, t=trace_ref)
