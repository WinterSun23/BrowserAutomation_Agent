from __future__ import annotations

import json
import re
from dataclasses import replace

from .db import GraphStore, profile_key
from .embed import cosine
from .ids import gid, now_ms, ulid
from .llm import complete_json
from .policy import ExplorationPolicy
from .schema import ensure_profile_schema
from .vault import Vault


SENSITIVE_TEXT = re.compile(
    r"(password|passwd|passcode|api[ _-]?key|secret|token|credit card|card number|ssn|\b[\w.+-]+@[\w-]+\.[\w.]+|\d{9,})",
    re.I)


class Profile:
    def __init__(self, store: GraphStore, uid: str, embedder, dim: int = 384):
        self.store, self.uid, self.embedder = store, uid, embedder
        self.key = profile_key(uid)
        ensure_profile_schema(store, self.key, dim)
        self.vault = Vault(uid)
        self.policy_overrides: dict = {}   # config.yaml `policy:` block and command-line flags; never written to the graph

    def q(self, cypher: str, **params):
        return self.store.query(self.key, cypher, params)

    def set_pref(self, key: str, value, source: str = "explicit", confidence: float = 1.0) -> None:
        self.q("MERGE (p:Preference {key:$k}) SET p.value = $v, p.source = $s, p.confidence = $c, p.updated = $now",
               k=key, v=json.dumps(value), s=source, c=confidence, now=now_ms())

    def get_prefs(self) -> dict:
        return {k: json.loads(v) for k, v in self.q("MATCH (p:Preference) RETURN p.key, p.value")}

    def default_policy(self) -> ExplorationPolicy:
        return self._stored_default_policy().with_overrides(self.policy_overrides)

    def _stored_default_policy(self) -> ExplorationPolicy:
        rows = self.q("MATCH (p:ExplorationPolicy {is_default:true}) RETURN p.data LIMIT 1")
        if rows:
            return ExplorationPolicy.from_dict(json.loads(rows[0][0]))
        pol = ExplorationPolicy()
        self.save_policy(pol, is_default=True)
        return pol

    def save_policy(self, policy: ExplorationPolicy, is_default: bool = False, domain: str | None = None,
                    device: str | None = None) -> None:
        if is_default:
            self.q("MATCH (p:ExplorationPolicy) SET p.is_default = false")
        self.q("MERGE (p:ExplorationPolicy {name:$n}) SET p.data = $d, p.is_default = $isdef",
               n=policy.name, d=json.dumps(policy.to_dict()), isdef=is_default)
        if domain:
            site_gid = gid("site", domain, device or "desktop")
            self.q("MATCH (:ExplorationPolicy)-[old:APPLIES_TO]->(:SiteEntry {gid:$g}) DELETE old", g=site_gid)
            self.q("MATCH (p:ExplorationPolicy {name:$n}), (s:SiteEntry {gid:$g}) MERGE (p)-[:APPLIES_TO]->(s)",
                   n=policy.name, g=site_gid)

    def policy_for(self, domain: str, device: str = "desktop") -> ExplorationPolicy:
        rows = self.q("MATCH (p:ExplorationPolicy)-[:APPLIES_TO]->(:SiteEntry {gid:$g}) RETURN p.data LIMIT 1",
                      g=gid("site", domain, device))
        stored = ExplorationPolicy.from_dict(json.loads(rows[0][0])) if rows else self._stored_default_policy()
        return stored.with_overrides(self.policy_overrides)

    def add_deny_patterns(self, domain: str, device: str, patterns: list[str]) -> list[str]:
        """Merge skip rules (bootstrap, guided exploration) into this site's own policy, creating it from the default
        on first use. Patterns that do not compile and ones already there are left out; returns the ones added."""
        rows = self.q("MATCH (p:ExplorationPolicy)-[:APPLIES_TO]->(:SiteEntry {gid:$g}) RETURN p.data LIMIT 1",
                      g=gid("site", domain, device))
        stored = ExplorationPolicy.from_dict(json.loads(rows[0][0])) if rows else self._stored_default_policy()
        added = []
        for p in patterns:
            try:
                re.compile(p)
            except (re.error, TypeError):
                continue
            if p and p not in stored.deny_patterns and p not in added:
                added.append(p)
        if added:
            # always a policy of this site's own, so a named policy shared with other sites is not changed
            self.save_policy(replace(stored, name=f"site:{domain}:{device}",deny_patterns=stored.deny_patterns + added),
                             domain=domain, device=device)
        return added

    def start_session(self) -> str:
        sid = ulid("ses")
        self.q("CREATE (:Session {id:$id, started:$now})", id=sid, now=now_ms())
        return sid

    def log_request(self, session_id: str, text: str, outcome: str, ref_graph: str | None = None,
                    ref_gid: str | None = None, kind: str | None = None, meta: dict | None = None) -> str:
        g = ulid("req")
        v = self.embedder.embed([text])[0]
        self.q("""MATCH (s:Session {id:$sid})
                  CREATE (s)-[:INCLUDED]->(:Request {gid:$g, text:$t, emb:vecf32($v), ts:$now, outcome:$o,
                                                    ref_graph:$rg, ref_gid:$rid, kind:$k, meta:$m})""",
               sid=session_id, g=g, t=text, v=v, now=now_ms(), o=outcome, rg=ref_graph, rid=ref_gid, k=kind,
               m=json.dumps(meta or {}))
        return g

    def set_secret(self, var: str, value: str, scope: str = "*") -> None:
        """The value goes to the vault; the graph keeps only the variable name, its scope and the vault key."""
        vault_key = f"{self.uid}/{scope}/{var}"
        self.vault.set(vault_key, value)
        self.q("MERGE (v:VaultRef {var:$var, scope:$scope}) SET v.vault_key = $k", var=var, scope=scope, k=vault_key)

    def secret_for(self, var: str, domain: str) -> str | None:
        for scope in (domain, "*"):
            rows = self.q("MATCH (v:VaultRef {var:$var, scope:$s}) RETURN v.vault_key", var=var, s=scope)
            if rows:
                return self.vault.get(rows[0][0])
        return None

    def secret_vars(self) -> list[dict]:
        return [{"var": v, "scope": s} for v, s in self.q("MATCH (v:VaultRef) RETURN v.var, v.scope ORDER BY v.var")]

    def infer_demo_format(self) -> None:
        if self.q("MATCH (p:Preference {key:'demo.format'}) WHERE p.source = 'explicit' RETURN count(p)")[0][0]:
            return
        rows = self.q("MATCH (r:Request {kind:'demo', outcome:'done'}) RETURN r.meta ORDER BY r.ts DESC LIMIT 5")
        firsts = [(json.loads(m or "{}").get("formats") or [None])[0] for (m,) in rows]
        firsts = [f for f in firsts if f]
        if firsts:
            best = max(set(firsts), key=firsts.count)
            if firsts.count(best) >= 3:
                self.set_pref("demo.format", best, source="inferred", confidence=0.6)

    def end_session(self, session_id: str, llm) -> None:
        texts = [t for (t,) in self.q("MATCH (:Session {id:$s})-[:INCLUDED]->(r:Request) RETURN r.text ORDER BY r.ts",
                                      s=session_id)]
        if not texts:
            return
        data = complete_json(llm, "You summarise a user's session with a web automation agent.",
                             "Requests:\n" + "\n".join(f"- {t}" for t in texts) + '\nReturn {"summary": "<one sentence>"}',
                             default={"summary": "; ".join(texts)[:300]})
        summary = str(data.get("summary") if isinstance(data, dict) else "")[:300] or "; ".join(texts)[:300]
        self.q("MATCH (s:Session {id:$s}) SET s.summary = $sum, s.ended = $now, s.emb = vecf32($v)",
               s=session_id, sum=summary, now=now_ms(), v=self.embedder.embed([summary])[0])

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
