from __future__ import annotations

import json

from .db import GraphStore, ops_key
from .embed import cosine
from .ids import gid, now_ms, ulid
from .normalize import url_template
from .schema import ensure_ops_schema


class Ops:
    def __init__(self, store: GraphStore, uid: str, domain: str):
        self.store, self.key = store, ops_key(uid, domain)
        ensure_ops_schema(store, self.key, 384)

    def q(self, cypher: str, **params):
        return self.store.query(self.key, cypher, params)

    def register_agent(self, agent_id: str, role: str, device: str | None = None) -> None:
        self.q("MERGE (a:Agent {id:$id}) SET a.role = coalesce(a.role, $r), a.device = coalesce(a.device, $d), a.status = 'idle', a.heartbeat = $now",
               id=agent_id, r=role, d=device, now=now_ms())

    def create_task(self, kind: str, request: str, device: str | None = None, priority: int = 0,
                    parent: str | None = None, policy: dict | None = None,
                    depends_on: list[str] | None = None) -> str:
        tid = ulid("task")
        self.q("CREATE (:Task {id:$id, kind:$k, request:$r, device:$d, priority:$p, status:'queued', created:$now, "
               "policy:$pol})", id=tid, k=kind, r=request, d=device, p=priority, now=now_ms(),
               pol=json.dumps(policy) if policy else None)
        if parent:
            self.q("MATCH (c:Task {id:$c}), (p:Task {id:$p}) CREATE (c)-[:SUBTASK_OF]->(p)", c=tid, p=parent)
        for d in depends_on or []:
            self.q("MATCH (c:Task {id:$c}), (d:Task {id:$d}) CREATE (c)-[:DEPENDS_ON]->(d)", c=tid, d=d)
        return tid

    def assign(self, task_id: str, agent_id: str) -> None:
        self.q("MATCH (t:Task {id:$t}), (a:Agent {id:$a}) MERGE (t)-[:ASSIGNED_TO]->(a) SET t.status = 'claimed'",
               t=task_id, a=agent_id)

    def set_status(self, task_id: str, status: str, result: dict | None = None) -> None:
        self.q("MATCH (t:Task {id:$t}) SET t.status = $s, t.updated = $now, t.result = $r",
               t=task_id, s=status, now=now_ms(), r=json.dumps(result) if result is not None else None)

    def get_task(self, task_id: str) -> dict:
        rows = self.q("MATCH (t:Task {id:$t}) OPTIONAL MATCH (t)-[:ASSIGNED_TO]->(a:Agent) "
                      "RETURN t.id, t.kind, t.request, t.status, t.result, a.id, t.created", t=task_id)
        r = rows[0]
        return dict(id=r[0], kind=r[1], request=r[2], status=r[3], result=json.loads(r[4]) if r[4] else None,
                    agent=r[5], created=r[6])

    def list_tasks(self, limit: int = 50) -> list[dict]:
        rows = self.q("MATCH (t:Task) RETURN t.id ORDER BY t.created DESC LIMIT $n", n=limit)
        return [self.get_task(r[0]) for r in rows]

    def claim(self, target_gid: str, graph: str, agent_id: str, ttl_ms: int = 60000) -> bool:
        now = now_ms()
        self.q("MATCH (c:Claim {target_gid:$g}) WHERE c.expires_at < $now DETACH DELETE c", g=target_gid, now=now)
        rows = self.q("""MATCH (a:Agent {id:$a})
                         OPTIONAL MATCH (c:Claim {target_gid:$g})
                         WITH a, c WHERE c IS NULL
                         CREATE (:Claim {target_gid:$g, graph:$graph, expires_at:$exp})-[:HELD_BY]->(a)
                         RETURN true""", a=agent_id, g=target_gid, graph=graph, exp=now + ttl_ms)
        return bool(rows)

    def release(self, target_gid: str, agent_id: str) -> None:
        self.q("MATCH (c:Claim {target_gid:$g})-[:HELD_BY]->(:Agent {id:$a}) DETACH DELETE c", g=target_gid, a=agent_id)

    def renew(self, target_gid: str, agent_id: str, ttl_ms: int = 120000) -> None:
        self.q("MATCH (c:Claim {target_gid:$g})-[:HELD_BY]->(:Agent {id:$a}) SET c.expires_at = $exp",
               g=target_gid, a=agent_id, exp=now_ms() + ttl_ms)

    # ---------- shared exploration work items ----------
    # kind 'page'  : load `url` and scan it (page-level work)
    # kind 'click' : try element `el_gid` on `url` after replaying the clicks in `pre` (path-level work)
    def add_item(self, scope: str, graph: str, kind: str, url: str, depth: int, el_gid: str | None = None,
                 pre=(), pre_cstates=(), priority: float = 0.0) -> str:
        g = gid("wk", scope, graph, kind, url_template(url), el_gid or "", *pre)
        self.q("""MERGE (w:WorkItem {gid:$g})
                  SET w.scope = $scope, w.graph = $graph, w.kind = $kind, w.url = coalesce(w.url, $url),
                      w.depth = coalesce(w.depth, $depth), w.el_gid = $el, w.pre = $pre, w.pre_cstates = $pcs,
                      w.status = coalesce(w.status, 'open'), w.attempts = coalesce(w.attempts, 0),
                      w.created = coalesce(w.created, $now), w.priority = coalesce(w.priority, $prio)""",
               g=g, scope=scope, graph=graph, kind=kind, url=url, depth=depth, el=el_gid, pre=list(pre),
               pcs=list(pre_cstates), now=now_ms(), prio=float(priority))
        return g

    def next_item(self, scope: str, agent_id: str, graph: str, ttl_ms: int = 120000,
                  min_priority: float | None = None) -> tuple[dict | None, bool]:
        """Claim the next open item (with `min_priority`, only items of at least that priority: guided rounds).
        Returns (item, others_hold_open_items)."""
        rows = self.q("MATCH (w:WorkItem {scope:$s, status:'open'}) "
                      "WHERE $mp IS NULL OR coalesce(w.priority, 0.0) >= $mp "
                      "RETURN w.gid, w.kind, w.url, w.depth, w.el_gid, w.pre, w.pre_cstates "
                      "ORDER BY coalesce(w.priority, 0.0) DESC, w.depth, w.created, w.gid LIMIT 50", s=scope,
                      mp=None if min_priority is None else float(min_priority))
        if not rows:
            return None, False
        for g, kind, url, depth, el, pre, pcs in rows:
            if not self.claim(g, graph, agent_id, ttl_ms):
                continue
            if self.q("MATCH (w:WorkItem {gid:$g, status:'open'}) RETURN count(w)", g=g)[0][0]:
                return dict(gid=g, kind=kind, url=url, depth=depth, el_gid=el, pre=tuple(pre or ()),
                            pre_cstates=tuple(pcs or ())), True
            self.release(g, agent_id)  # finished by someone else between our read and our claim
        return None, True

    def open_items(self, scope: str, limit: int = 5000) -> list[dict]:
        """Open work items of a scope, most promising first (what guided exploration groups into a frontier)."""
        rows = self.q("MATCH (w:WorkItem {scope:$s, status:'open'}) "
                      "RETURN w.gid, w.kind, w.url, w.depth, w.el_gid, coalesce(w.priority, 0.0), w.pre "
                      "ORDER BY coalesce(w.priority, 0.0) DESC, w.depth, w.created, w.gid LIMIT $n", s=scope, n=limit)
        return [dict(gid=r[0], kind=r[1], url=r[2], depth=r[3], el_gid=r[4], priority=r[5], pre=tuple(r[6] or ()))
                for r in rows]

    def set_priorities(self, gids: list[str], priority: float) -> None:
        if gids:
            self.q("MATCH (w:WorkItem) WHERE w.gid IN $g AND w.status = 'open' SET w.priority = $p",
                   g=list(gids), p=float(priority))

    def bump_priorities(self, gids: list[str], amount: float) -> None:
        if gids:
            self.q("MATCH (w:WorkItem) WHERE w.gid IN $g AND w.status = 'open' "
                   "SET w.priority = coalesce(w.priority, 0.0) + $a", g=list(gids), a=float(amount))

    def close_items(self, gids: list[str], reason: str) -> None:
        """Mark open items done without working on them (skipped by the planner)."""
        if gids:
            self.q("MATCH (w:WorkItem) WHERE w.gid IN $g AND w.status = 'open' SET w.status = 'done', w.error = $e",
                   g=list(gids), e=reason)

    def reseed(self, scope: str, graph: str, url: str, priority: float = 100.0) -> str:
        """Queue `url` as a page item of depth 0 (or bring an open one back to depth 0) with a high priority, so the
        next round starts there with its full depth budget. A page already done is not loaded again."""
        g = self.add_item(scope, graph, "page", url, 0, priority=priority)
        self.q("MATCH (w:WorkItem {gid:$g}) WHERE w.status = 'open' SET w.depth = 0, w.priority = $p",
               g=g, p=float(priority))
        return g

    def backlog(self, scope: str) -> int:
        """Open work items nobody holds a claim on (what an extra explorer could start on right now)."""
        rows = self.q("MATCH (w:WorkItem {scope:$s, status:'open'}) OPTIONAL MATCH (c:Claim {target_gid:w.gid}) "
                      "WITH w, c WHERE c IS NULL RETURN count(w)", s=scope)
        return int(rows[0][0]) if rows else 0

    def finish_item(self, item_gid: str, agent_id: str, error: str | None = None) -> None:
        self.q("MATCH (w:WorkItem {gid:$g}) SET w.status = 'done', w.done_by = $a, w.error = $e, "
               "w.attempts = coalesce(w.attempts, 0) + 1", g=item_gid, a=agent_id, e=error)
        self.release(item_gid, agent_id)

    def item_stats(self, scope: str) -> dict:
        rows = self.q("MATCH (w:WorkItem {scope:$s}) RETURN w.status, count(w), max(w.attempts)", s=scope)
        return {r[0]: {"count": r[1], "max_attempts": r[2]} for r in rows}

    # ---------- coordination: dependencies, heartbeats, notes, handoffs ----------
    def runnable(self, limit: int = 20) -> list[dict]:
        """Queued tasks whose dependencies are all done, highest priority first."""
        out = []
        for (tid,) in self.q("MATCH (t:Task {status:'queued'}) RETURN t.id ORDER BY t.priority DESC, t.created"):
            deps = self.q("MATCH (:Task {id:$t})-[:DEPENDS_ON]->(d:Task) RETURN d.status", t=tid)
            if all(s == "done" for (s,) in deps):
                out.append(self.get_task(tid))
            if len(out) >= limit:
                break
        return out

    def heartbeat(self, agent_id: str, status: str) -> None:
        self.q("MATCH (a:Agent {id:$a}) SET a.status = $s, a.heartbeat = $now", a=agent_id, s=status, now=now_ms())

    def add_note(self, text: str, agent_id: str, embedder=None, ref_graph: str | None = None,
                 ref_gid: str | None = None) -> str:
        """A short fact one agent leaves for the others (a login wall, a robots block); searchable by meaning."""
        g = ulid("note")
        self.q("MERGE (a:Agent {id:$a}) CREATE (n:Note {gid:$g, text:$t, ts:$now, ref_graph:$rg, ref_gid:$rid})"
               "-[:BY]->(a)", a=agent_id, g=g, t=text, now=now_ms(), rg=ref_graph, rid=ref_gid)
        if embedder is not None:
            self.q("MATCH (n:Note {gid:$g}) SET n.emb = vecf32($v)", g=g, v=embedder.embed([text])[0])
        return g

    def notes(self, text: str, embedder, k: int = 5) -> list[tuple[str, float]]:
        v = embedder.embed([text])[0]
        rows = self.q("CALL db.idx.vector.queryNodes('Note', 'emb', $k, vecf32($v)) YIELD node "
                      "RETURN node.text, node.emb", k=k, v=v)
        return sorted(((t, cosine(v, list(e))) for t, e in rows), key=lambda r: r[1], reverse=True)

    def record_handoff(self, from_agent: str, to_agent: str, task_id: str | None, reason: str, context: dict) -> str:
        g = ulid("ho")
        self.q("""MERGE (f:Agent {id:$f}) MERGE (t:Agent {id:$t})
                  CREATE (h:Handoff {gid:$g, reason:$r, context:$c, ts:$now}), (h)-[:FROM_AGENT]->(f),
                         (h)-[:TO_AGENT]->(t)""", f=from_agent, t=to_agent, g=g, r=reason,
               c=json.dumps(context), now=now_ms())
        if task_id:
            self.q("MATCH (h:Handoff {gid:$g}), (k:Task {id:$t}) CREATE (h)-[:FOR_TASK]->(k)", g=g, t=task_id)
        return g
