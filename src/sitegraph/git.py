"""Git for the device graph (spec §6.1-6.3, §12.10): commits are lists of recorded `Change`s kept in the ops graph, a
branch is a `GRAPH.COPY` of the device graph, main is the device graph itself. Merge, revert, checkpoints and compaction
are added on top of this core by later tasks."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field

from .db import device_key, ops_key
from .graph.map_writer import MapWriter
from .graph.recording import STRIP_ACTIVITY, Change, RecordingMapWriter, copy_activity, replay
from .ids import now_ms, ulid
from .ops import Ops
from .schema import ensure_ops_schema

_COPY_LOCK = threading.Lock()      # GRAPH.COPY forks the server: concurrent copies can fail with 'could not fork'

OVERWRITE_OPS = {"bump_locator", "quarantine", "unquarantine", "set_props", "restore_node", "delete_node",
                 "restore_edge", "delete_edge", "set_cstate_shot", "set_cstate_snapshot", "rename_state"}


@dataclass
class BranchInfo:
    name: str
    graph_key: str
    device: str
    base_commit: str
    owner: str


@dataclass
class MergeResult:
    commit: str | None
    applied: int = 0
    skipped: int = 0
    conflicts: list = field(default_factory=list)


class MergeBusy(Exception):
    pass


class GitOps:
    def __init__(self, store, uid: str, domain: str, settings):
        self.store, self.uid, self.domain, self.s = store, uid, domain, settings
        self.key = ops_key(uid, domain)
        ensure_ops_schema(store, self.key, settings.embed_dim)

    def q(self, cypher: str, **params):
        return self.store.query(self.key, cypher, params)

    def ro(self, cypher: str, **params):
        return self.store.ro_query(self.key, cypher, params)

    def main_key(self, device: str) -> str:
        return device_key(self.uid, self.domain, device)

    def _copy(self, src: str, dst: str, tries: int = 40) -> None:
        """GRAPH.COPY, one at a time in this process and retried when the server cannot fork yet."""
        with _COPY_LOCK:
            for attempt in range(tries):
                try:
                    self.store.copy(src, dst)
                    return
                except Exception as exc:
                    if "fork" not in str(exc).lower() or attempt == tries - 1:
                        raise
                    time.sleep(0.25)

    # ---------- commits & branches ----------
    def _branch_head(self, name: str) -> str | None:
        rows = self.ro("MATCH (:Branch {name:$n})-[:HEAD]->(c:Commit) RETURN c.id", n=name)
        return rows[0][0] if rows else None

    def _move_head(self, name: str, cid: str) -> None:
        self.q("MATCH (:Branch {name:$n})-[h:HEAD]->() DELETE h", n=name)
        self.q("MATCH (b:Branch {name:$n}), (c:Commit {id:$c}) CREATE (b)-[:HEAD]->(c)", n=name, c=cid)

    def _new_commit(self, parents: list[str], msg: str, agent: str, device: str, graph_key: str,
                    changes: list[Change], link_existing: list[tuple[str, bool]] | None = None) -> str:
        digests = [c.digest() for c in changes]
        seed = "|".join(parents + digests + [str(now_ms()), os.urandom(4).hex()])
        cid = "c:" + hashlib.sha1(seed.encode()).hexdigest()[:16]
        n = len(changes) + len(link_existing or [])
        self.q("CREATE (:Commit {id:$c, msg:$m, ts:$now, device:$d, graph_key:$k, stats:$st})",
               c=cid, m=msg, now=now_ms(), d=device, k=graph_key, st=json.dumps({"changes": n}))
        for i, p in enumerate(parents):
            self.q("MATCH (c:Commit {id:$c}), (p:Commit {id:$p}) CREATE (c)-[:PARENT {order:$i}]->(p)", c=cid, p=p, i=i)
        self.q("MERGE (a:Agent {id:$a}) WITH a MATCH (c:Commit {id:$c}) CREATE (c)-[:AUTHORED_BY]->(a)", a=agent, c=cid)
        if changes:
            rows = [{**ch.to_props(), "gid": ulid("chg")} for ch in changes]
            self.q("""UNWIND $rows AS r MATCH (c:Commit {id:$c})
                      CREATE (c)-[:HAS_CHANGE {applied:true}]->(:Change {gid:r.gid, seq:r.seq, op:r.op, args:r.args,
                                                                         touched:r.touched, ts:r.ts})""",
                   rows=rows, c=cid)
        for xg, applied in link_existing or []:
            self.q("MATCH (c:Commit {id:$c}), (x:Change {gid:$x}) CREATE (c)-[:HAS_CHANGE {applied:$a}]->(x)",
                   c=cid, x=xg, a=applied)
        return cid

    def init_main(self, device: str, agent_id: str = "system") -> str:
        """The genesis commit of a device graph (and its `genesis` tag); idempotent."""
        name = f"{device}/main"
        head = self._branch_head(name)
        if head:
            return head
        self.store.query(self.main_key(device), "RETURN 1")
        cid = self._new_commit([], "genesis", agent_id, device, self.main_key(device), [])
        self.q("MERGE (b:Branch {name:$n}) SET b.graph_key = $k, b.device = $d, b.status = 'open', b.created = $now, "
               "b.base_commit = $c", n=name, k=self.main_key(device), d=device, now=now_ms(), c=cid)
        self._move_head(name, cid)
        self.tag(device, "genesis")
        return cid

    def head(self, device: str) -> str:
        return self._branch_head(f"{device}/main") or self.init_main(device)

    def fork(self, device: str, agent_id: str, task_id: str | None = None) -> BranchInfo:
        """A private copy of main (map only, no activity) for one agent to work on."""
        base = self.head(device)
        short = os.urandom(4).hex()
        name = f"{device}/{agent_id}-{short}"
        gkey = f"{self.main_key(device)}@br:{agent_id}-{short}"
        self._copy(self.main_key(device), gkey)
        self.store.query(gkey, STRIP_ACTIVITY)
        self.q("""MERGE (a:Agent {id:$a})
                  CREATE (b:Branch {name:$n, graph_key:$k, device:$d, base_commit:$base, status:'open', created:$now})
                         -[:OWNED_BY]->(a)""", a=agent_id, n=name, k=gkey, d=device, base=base, now=now_ms())
        self._move_head(name, base)
        if task_id:
            self.q("MATCH (b:Branch {name:$n}), (t:Task {id:$t}) CREATE (b)-[:FOR_TASK]->(t)", n=name, t=task_id)
        return BranchInfo(name, gkey, device, base, agent_id)

    def branch(self, name: str) -> BranchInfo:
        r = self.ro("MATCH (b:Branch {name:$n})-[:OWNED_BY]->(a:Agent) RETURN b.graph_key, b.device, b.base_commit, a.id",
                    n=name)[0]
        return BranchInfo(name, r[0], r[1], r[2], r[3])

    def recorder(self, br: BranchInfo) -> RecordingMapWriter:
        return RecordingMapWriter(self.store, br.graph_key, self.s)

    def main_recorder(self, device: str) -> RecordingMapWriter:
        self.init_main(device)
        return RecordingMapWriter(self.store, self.main_key(device), self.s)

    def commit(self, br: BranchInfo, writer: RecordingMapWriter, agent_id: str, msg: str) -> str | None:
        """Save the writer's recorded changes as a commit on the branch (None when nothing changed)."""
        if not writer.changes:
            return None
        cid = self._new_commit([self._branch_head(br.name)], msg, agent_id, br.device, br.graph_key, writer.changes)
        self._move_head(br.name, cid)
        writer.changes = []
        return cid

    def commit_main(self, device: str, writer: RecordingMapWriter, agent_id: str, msg: str) -> str | None:
        if not writer.changes:
            return None
        cid = self._new_commit([self.head(device)], msg, agent_id, device, self.main_key(device), writer.changes)
        self._move_head(f"{device}/main", cid)
        writer.changes = []
        self._maybe_checkpoint(device)
        return cid

    # ---------- history ----------
    def first_parent_chain(self, cid: str, stop: str | None = None, limit: int = 100000) -> list[str]:
        chain, cur = [], cid
        while cur and cur != stop and len(chain) < limit:
            chain.append(cur)
            rows = self.ro("MATCH (:Commit {id:$c})-[:PARENT {order:0}]->(p:Commit) RETURN p.id", c=cur)
            cur = rows[0][0] if rows else None
        return chain

    def log(self, device: str, limit: int = 50) -> list[dict]:
        out = []
        for cid in self.first_parent_chain(self.head(device), limit=limit):
            r = self.ro("""MATCH (c:Commit {id:$c}) OPTIONAL MATCH (c)-[:AUTHORED_BY]->(a:Agent)
                           OPTIONAL MATCH (t:Tag)-[:AT]->(c)
                           RETURN c.msg, c.ts, a.id, collect(t.name), c.stats, c.compacted""", c=cid)[0]
            out.append({"id": cid, "msg": r[0], "ts": r[1], "agent": r[2], "tags": r[3],
                        "stats": json.loads(r[4] or "{}"), "compacted": bool(r[5])})
        return out

    def changes_of(self, cid: str, applied_only: bool = True) -> list[tuple[str, Change]]:
        rows = self.ro("""MATCH (:Commit {id:$c})-[h:HAS_CHANGE]->(x:Change)
                          WHERE (NOT $only) OR h.applied RETURN x.gid, properties(x) ORDER BY x.ts, x.seq""",
                       c=cid, only=applied_only)
        return [(g, Change.from_props(p)) for g, p in rows]

    def blame(self, gid_: str) -> list[dict]:
        """Every commit (with its agent) whose changes touched this entity, oldest first."""
        rows = self.ro("""MATCH (c:Commit)-[h:HAS_CHANGE]->(x:Change) WHERE x.touched CONTAINS $g
                          OPTIONAL MATCH (c)-[:AUTHORED_BY]->(a:Agent)
                          RETURN c.id, c.msg, c.ts, a.id, x.op, h.applied ORDER BY c.ts""", g=gid_)
        return [{"commit": r[0], "msg": r[1], "ts": r[2], "agent": r[3], "op": r[4], "applied": r[5]} for r in rows]

    def handoff(self, br: BranchInfo, from_agent: str, to_agent: str, task_id: str | None = None,
                reason: str = "", context: dict | None = None) -> BranchInfo:
        """Pass a branch (and what the sender knows) to another agent."""
        Ops(self.store, self.uid, self.domain).record_handoff(from_agent, to_agent, task_id, reason,
                                                              {"branch": br.name, **(context or {})})
        self.q("MATCH (b:Branch {name:$n})-[o:OWNED_BY]->() DELETE o", n=br.name)
        self.q("MERGE (a:Agent {id:$a}) WITH a MATCH (b:Branch {name:$n}) CREATE (b)-[:OWNED_BY]->(a)",
               a=to_agent, n=br.name)
        return BranchInfo(br.name, br.graph_key, br.device, br.base_commit, to_agent)

    # ---------- merge ----------
    def _touched_since(self, head: str, base: str) -> dict[str, int]:
        touched: dict[str, int] = {}
        for cid in self.first_parent_chain(head, stop=base):
            for _, ch in self.changes_of(cid):
                for t in ch.touched:
                    touched[t["gid"]] = max(touched.get(t["gid"], 0), ch.ts)
        return touched

    def _lock(self, device: str, agent_id: str, wait_s: int) -> Ops:
        """The merge lock: a Claim on `merge:{device}`, so only one writer at a time changes a device main."""
        ops = Ops(self.store, self.uid, self.domain)
        ops.register_agent(agent_id, "curator")
        deadline = time.monotonic() + wait_s
        while not ops.claim(f"merge:{device}", self.key, agent_id, ttl_ms=600_000):
            if time.monotonic() >= deadline:
                raise MergeBusy(f"another merge into {device}/main is running")
            time.sleep(0.5)
        return ops

    def merge(self, br: BranchInfo, agent_id: str = "curator", recompute=None, keep_branch: bool = False,
              wait_s: int = 120) -> MergeResult:
        """Replay the branch's changes onto main. Overwrite-type changes to something main also changed since the
        fork are conflicts, resolved by the newer observation; additive changes (counters, new entities) always apply,
        so two branches that saw the same screen never duplicate it and never double-count."""
        ops = self._lock(br.device, agent_id, wait_s)
        try:
            main_head = self.head(br.device)
            branch_head = self._branch_head(br.name)
            main_touched = self._touched_since(main_head, br.base_commit)
            writer = MapWriter(self.store, self.main_key(br.device), self.s)
            result, links = MergeResult(None), []
            for cid in reversed(self.first_parent_chain(branch_head, stop=br.base_commit)):
                for xg, ch in self.changes_of(cid):
                    hit = [t["gid"] for t in ch.touched if t["gid"] in main_touched]
                    if ch.op in OVERWRITE_OPS and hit:
                        newer = ch.ts >= max(main_touched[g] for g in hit)
                        result.conflicts.append({"gid": hit[0], "field": ch.op, "theirs": json.dumps(ch.args)[:500],
                                                 "rule": "newer-observation-wins",
                                                 "resolution": "theirs" if newer else "ours"})
                        if not newer:
                            result.skipped += 1
                            links.append((xg, False))
                            continue
                    replay(writer, ch)
                    result.applied += 1
                    links.append((xg, True))
            copy_activity(self.store, br.graph_key, self.main_key(br.device))
            mcid = self._new_commit([main_head, branch_head], f"merge {br.name}", agent_id, br.device,
                                    self.main_key(br.device), [], link_existing=links)
            for c in result.conflicts:
                self.q("MATCH (m:Commit {id:$m}) CREATE (:Conflict {gid:$g, field:$f, theirs:$t, rule:$r, "
                       "resolution:$res})-[:IN_MERGE]->(m)", m=mcid, g=c["gid"], f=c["field"], t=c["theirs"],
                       r=c["rule"], res=c["resolution"])
            self._move_head(f"{br.device}/main", mcid)
            self.q("MATCH (b:Branch {name:$n}) SET b.status = 'merged', b.merged_into = $m", n=br.name, m=mcid)
            result.commit = mcid
        finally:
            ops.release(f"merge:{br.device}", agent_id)
        if recompute is not None:
            recompute()
        if not keep_branch:
            self.store.delete(br.graph_key)
        self._maybe_checkpoint(br.device)
        return result

    # ---------- tags, checkpoints, history ----------
    def tag(self, device: str, name: str) -> str:
        key = f"{self.main_key(device)}@tag:{name}"
        if self.store.exists(key):
            raise ValueError(f"tag {name} already exists")
        self._copy(self.main_key(device), key)
        head = self._branch_head(f"{device}/main")
        self.q("CREATE (:Tag {name:$n, graph_key:$k, device:$d, ts:$now})", n=f"{device}/{name}", k=key, d=device,
               now=now_ms())
        if head:
            self.q("MATCH (t:Tag {name:$n}), (c:Commit {id:$c}) CREATE (t)-[:AT]->(c)", n=f"{device}/{name}", c=head)
        return key

    def tags(self, device: str) -> list[dict]:
        rows = self.ro("MATCH (t:Tag {device:$d}) OPTIONAL MATCH (t)-[:AT]->(c:Commit) "
                       "RETURN t.name, t.graph_key, c.id, t.ts ORDER BY t.ts", d=device)
        return [{"name": r[0], "graph_key": r[1], "commit": r[2], "ts": r[3]} for r in rows]

    def _tag_at(self, cid: str) -> str | None:
        rows = self.ro("MATCH (t:Tag)-[:AT]->(:Commit {id:$c}) RETURN t.graph_key ORDER BY t.ts DESC LIMIT 1", c=cid)
        return rows[0][0] if rows else None

    def _maybe_checkpoint(self, device: str) -> None:
        """A snapshot tag every `checkpoint_every` commits, so checkout never replays more than that."""
        since = 0
        for cid in self.first_parent_chain(self.head(device)):
            if self._tag_at(cid):
                break
            since += 1
        if since >= self.s.checkpoint_every:
            self.tag(device, f"auto-{self.head(device)[2:10]}")

    def revert(self, device: str, commit_id: str, agent_id: str = "curator") -> str:
        """Undo a commit from the prior-state snapshots of its changes (history stays; the revert is itself a commit
        and can be reverted)."""
        writer = self.main_recorder(device)
        for _, ch in reversed(self.changes_of(commit_id)):
            # nodes before edges: an edge can only be restored once both of its endpoints exist again
            for t in sorted(reversed(ch.touched), key=lambda x: x["kind"] != "node"):
                prev = t["prev"]
                if t["kind"] == "node":
                    if prev is None:
                        writer.delete_node(t["label"], t["gid"])
                    else:
                        writer.restore_node(t["label"], t["gid"],
                                            {k: v for k, v in prev.items() if k not in ("_rels", "_labels")})
                        for direction, rtype, olabel, ogid, rprops in prev.get("_rels", []):
                            writer.restore_rel(t["label"], t["gid"], direction, rtype, olabel, ogid, rprops or {})
                        for extra in prev.get("_labels", []):
                            if extra in ("Entry", "Blocking", "Frontier"):
                                writer.q(f"MATCH (n:{t['label']} {{gid:$g}}) SET n:{extra}", g=t["gid"])
                elif prev is None:
                    writer.delete_edge(t["type"], t["gid"])
                else:
                    writer.restore_edge(t["type"], t["gid"], prev["src"], prev["dst"], prev["props"])
        cid = self._new_commit([self.head(device)], f"revert {commit_id}", agent_id, device, self.main_key(device),
                               writer.changes)
        self._move_head(f"{device}/main", cid)
        return cid

    def checkout(self, device: str, commit_id: str) -> str:
        """A read-only graph as it was at `commit_id`: the nearest snapshot at or before it, plus the later commits."""
        base_tag, pending = None, []
        for cid in self.first_parent_chain(commit_id):
            base_tag = self._tag_at(cid)
            if base_tag:
                break
            pending.append(cid)
        if base_tag is None:
            raise ValueError(f"no snapshot at or before {commit_id}")
        key = f"{self.main_key(device)}@at:{commit_id[2:10]}"
        self.store.delete(key)
        self._copy(base_tag, key)
        writer = MapWriter(self.store, key, self.s)
        for cid in reversed(pending):
            for _, ch in self.changes_of(cid):
                replay(writer, ch)
        return key

    def compact(self, device: str, keep_commits: int | None = None) -> int:
        """Drop Change payloads of commits older than the newest tagged commit outside the keep window."""
        keep = self.s.compact_keep_commits if keep_commits is None else keep_commits
        chain = self.first_parent_chain(self.head(device))
        older = chain[keep:]
        boundary = next((i for i, cid in enumerate(older) if self._tag_at(cid)), None)
        if boundary is None:
            return 0
        removed = 0
        for cid in older[boundary + 1:]:
            n = self.ro("MATCH (:Commit {id:$c})-[:HAS_CHANGE]->(x:Change) RETURN count(x)", c=cid)[0][0]
            self.q("MATCH (:Commit {id:$c})-[:HAS_CHANGE]->(x:Change) DETACH DELETE x", c=cid)
            self.q("MATCH (c:Commit {id:$c}) SET c.compacted = true", c=cid)
            removed += max(n, 1)
        return removed
