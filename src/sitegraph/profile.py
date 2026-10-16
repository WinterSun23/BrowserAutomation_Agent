from __future__ import annotations

import json

from .db import GraphStore, profile_key
from .embed import cosine
from .ids import gid, now_ms, ulid
from .policy import ExplorationPolicy
from .schema import ensure_profile_schema


class Profile:
    def __init__(self, store: GraphStore, uid: str, embedder, dim: int = 384):
        self.store, self.uid, self.embedder = store, uid, embedder
        self.key = profile_key(uid)
        ensure_profile_schema(store, self.key, dim)

    def q(self, cypher: str, **params):
        return self.store.query(self.key, cypher, params)

    def set_pref(self, key: str, value, source: str = "explicit", confidence: float = 1.0) -> None:
        self.q("MERGE (p:Preference {key:$k}) SET p.value = $v, p.source = $s, p.confidence = $c, p.updated = $now",
               k=key, v=json.dumps(value), s=source, c=confidence, now=now_ms())

    def get_prefs(self) -> dict:
        return {k: json.loads(v) for k, v in self.q("MATCH (p:Preference) RETURN p.key, p.value")}

    def default_policy(self) -> ExplorationPolicy:
        rows = self.q("MATCH (p:ExplorationPolicy {is_default:true}) RETURN p.data LIMIT 1")
        if rows:
            return ExplorationPolicy.from_dict(json.loads(rows[0][0]))
        pol = ExplorationPolicy()
        self.save_policy(pol, is_default=True)
        return pol

    def save_policy(self, policy: ExplorationPolicy, is_default: bool = False) -> None:
        if is_default:
            self.q("MATCH (p:ExplorationPolicy) SET p.is_default = false")
        self.q("MERGE (p:ExplorationPolicy {name:$n}) SET p.data = $d, p.is_default = $isdef",
               n=policy.name, d=json.dumps(policy.to_dict()), isdef=is_default)

    def start_session(self) -> str:
        sid = ulid("ses")
        self.q("CREATE (:Session {id:$id, started:$now})", id=sid, now=now_ms())
        return sid

    def log_request(self, session_id: str, text: str, outcome: str, ref_graph: str | None = None,
                    ref_gid: str | None = None) -> str:
        g = ulid("req")
        v = self.embedder.embed([text])[0]
        self.q("""MATCH (s:Session {id:$sid})
                  CREATE (s)-[:INCLUDED]->(:Request {gid:$g, text:$t, emb:vecf32($v), ts:$now, outcome:$o,
                                                    ref_graph:$rg, ref_gid:$rid})""",
               sid=session_id, g=g, t=text, v=v, now=now_ms(), o=outcome, rg=ref_graph, rid=ref_gid)
        return g

    def remember(self, text: str, kind: str = "semantic", importance: float = 0.5, ref_graph: str | None = None,
                 ref_gid: str | None = None) -> str:
        g = gid("mem", text)
        v = self.embedder.embed([text])[0]
        self.q("MERGE (m:Memory {gid:$g}) SET m.text = $t, m.kind = $k, m.importance = $i, m.emb = vecf32($v), "
               "m.ts = $now, m.ref_graph = $rg, m.ref_gid = $rid",
               g=g, t=text, k=kind, i=importance, v=v, now=now_ms(), rg=ref_graph, rid=ref_gid)
        return g

    def recall(self, text: str, k: int = 5) -> list[tuple[str, float]]:
        v = self.embedder.embed([text])[0]
        rows = self.store.ro_query(self.key, "CALL db.idx.vector.queryNodes('Memory', 'emb', $k, vecf32($v)) "
                                             "YIELD node RETURN node.text, node.emb", {"k": k, "v": v})
        return sorted(((t, cosine(v, list(e))) for t, e in rows), key=lambda r: r[1], reverse=True)

    def list_memories(self, limit: int = 50) -> list[dict]:
        rows = self.q("MATCH (m:Memory) RETURN m.gid, m.text, m.kind, m.ts ORDER BY m.ts DESC LIMIT $n", n=limit)
        return [dict(gid=r[0], text=r[1], kind=r[2], ts=r[3]) for r in rows]

    def forget(self, memory_gid: str) -> bool:
        self.q("MATCH (m:Memory {gid:$g}) DETACH DELETE m", g=memory_gid)
        return not self.q("MATCH (m:Memory {gid:$g}) RETURN count(m)", g=memory_gid)[0][0]

    def register_site(self, domain: str, device: str, graph_key: str, ops_graph_key: str, entry_url: str) -> None:
        self.q("""MERGE (s:SiteEntry {gid:$g})
                  SET s.domain = $d, s.device = $dev, s.graph_key = $k, s.ops_key = $ok, s.entry_url = $u,
                      s.last_used = $now""",
               g=gid("site", domain, device), d=domain, dev=device, k=graph_key, ok=ops_graph_key, u=entry_url,
               now=now_ms())

    def sites(self) -> list[dict]:
        rows = self.q("MATCH (s:SiteEntry) RETURN s.domain, s.device, s.graph_key, s.ops_key, s.entry_url "
                      "ORDER BY s.last_used DESC")
        return [dict(domain=r[0], device=r[1], key=r[2], ops_key=r[3], entry_url=r[4]) for r in rows]
