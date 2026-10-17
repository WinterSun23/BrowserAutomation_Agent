from __future__ import annotations

import json
from collections import defaultdict

from ..capture.segment import jaccard
from ..embed import cosine
from ..graph.queries import best_routes, entry_state
from ..ids import gid
from ..llm import complete_json
from ..schema import ensure_semantic_schema


def semantic_key(uid: str) -> str:
    return f"u:{uid}:semantic"


def _mean(vecs: list[list[float]]) -> list[float]:
    n = len(vecs)
    return [sum(v[i] for v in vecs) / n for i in range(len(vecs[0]))]


class Distiller:
    """Derives the cross-site / cross-device semantic graph from all device graphs (spec §8). Read-only on sources."""

    def __init__(self, store, uid: str, embedder, llm, settings, profile):
        self.store, self.uid, self.embedder, self.llm, self.s, self.profile = store, uid, embedder, llm, settings, profile
        self.key = semantic_key(uid)

    def q(self, cypher: str, **params):
        return self.store.query(self.key, cypher, params)

    def ro(self, key: str, cypher: str, **params):
        return self.store.ro_query(key, cypher, params)

    def _sites(self) -> list[dict]:
        return sorted(self.profile.sites(), key=lambda s: (s["domain"], s["device"]))

    def rebuild(self) -> dict:
        """Delete and rebuild the whole semantic graph from the device graphs. Idempotent: same inputs, same output."""
        self.store.delete(self.key)
        ensure_semantic_schema(self.store, self.key, self.s.embed_dim)
        stats: dict = {}
        stats.update(self._refs_and_uipatterns())
        stats.update(self._logical_states())
        stats.update(self._step_and_flow_patterns())
        stats.update(self._followed_by())
        self._algorithms()
        return stats

    # ---------- UIPatterns ----------
    def _refs_and_uipatterns(self) -> dict:
        comps = []
        for site in self._sites():
            self.q("MERGE (s:SiteRef {gid:$g}) SET s.domain = $d, s.graph_keys = coalesce(s.graph_keys, []) + [$k]",
                   g=gid("sr", site["domain"]), d=site["domain"], k=site["key"])
            for cg, kind, label, emb in self.ro(site["key"], "MATCH (k:Component) WHERE k.emb IS NOT NULL "
                                                             "RETURN k.gid, k.kind, coalesce(k.label, k.kind), k.emb "
                                                             "ORDER BY k.gid"):
                comps.append({"graph": site["key"], "device": site["device"], "domain": site["domain"], "gid": cg,
                              "kind": kind, "label": label, "emb": list(emb)})
        clusters: list[dict] = []
        for c in comps:
            best, best_sim = None, 0.0
            for cl in clusters:
                if cl["kind"] != c["kind"]:
                    continue
                sim = cosine(c["emb"], cl["centroid"])
                if sim >= self.s.uipattern_sim and sim > best_sim:
                    best, best_sim = cl, sim
            if best is None:
                clusters.append({"kind": c["kind"], "leader": c["label"], "members": [(c, 1.0)], "centroid": c["emb"]})
            else:
                best["members"].append((c, best_sim))
                best["centroid"] = _mean([m["emb"] for m, _ in best["members"]])
        for cl in clusters:
            labels = sorted({m["label"] for m, _ in cl["members"]})[:6]
            data = complete_json(self.llm, "You name recurring UI components.",
                                 f"Name this UI pattern in 1-3 words. Kind: {cl['kind']}. Examples: {labels}\n"
                                 'Return {"name": "..."}', default={"name": cl["kind"]})
            name = str(data.get("name") if isinstance(data, dict) else "") or cl["kind"]
            ug = gid("up", cl["kind"], cl["leader"])
            self.q("MERGE (u:UIPattern {gid:$g}) SET u.name = $n, u.kind = $k, u.support = $s, u.canonical_label = $l, "
                   "u.emb = vecf32($v)", g=ug, n=name, k=cl["kind"], l=cl["leader"],
                   s=len({m["domain"] for m, _ in cl["members"]}), v=cl["centroid"])
            for m, conf in cl["members"]:
                self.q("""MERGE (r:ComponentRef {gid:$rg}) SET r.ref_graph = $graph, r.ref_gid = $cg, r.device = $dev
                          WITH r MATCH (u:UIPattern {gid:$ug}), (s:SiteRef {gid:$sg})
                          MERGE (r)-[i:INSTANCE_OF]->(u) SET i.conf = $conf MERGE (r)-[:ON_SITE]->(s)""",
                       rg=gid("cr", m["graph"], m["gid"]), graph=m["graph"], cg=m["gid"], dev=m["device"], ug=ug,
                       sg=gid("sr", m["domain"]), conf=round(conf, 4))
        return {"ui_patterns": len(clusters), "component_refs": len(comps)}

    # ---------- LogicalStates ----------
    def _logical_states(self) -> dict:
        by_domain: dict[str, list[dict]] = defaultdict(list)
        for site in self._sites():
            by_domain[site["domain"]].append(site)
        total = aligned = 0
        for domain, sites in by_domain.items():
            states: dict[str, list[dict]] = {}
            for site in sites:
                rows = self.ro(site["key"], """MATCH (p:Page)-[:HAS_STATE]->(s:State)
                    OPTIONAL MATCH (i:Intent)-[:FULFILLED_BY]->(s)
                    RETURN s.gid, p.url_template, s.emb, coalesce(s.summary, ''), collect(i.text) ORDER BY s.gid""")
                states[site["device"]] = [{"graph": site["key"], "gid": g, "tpl": t, "emb": list(e) if e else None,
                                           "summary": su, "intents": set(it)} for g, t, e, su, it in rows]
            devices = sorted(states, key=lambda d: (d != "desktop", d))
            base = devices[0]
            logical = [{base: s} for s in states[base]]
            for dev in devices[1:]:
                for s in states[dev]:
                    best, best_score = None, 0.0
                    for L in logical:
                        if dev in L or base not in L:
                            continue
                        b = L[base]
                        score = 0.5 * (s["tpl"] == b["tpl"]) + 0.3 * jaccard(s["intents"], b["intents"])
                        if s["emb"] and b["emb"]:
                            score += 0.2 * cosine(s["emb"], b["emb"])
                        if score >= self.s.logical_state_min and score > best_score:
                            best, best_score = L, score
                    if best is not None:
                        best[dev] = s
                    else:
                        logical.append({dev: s})
            for L in logical:
                members = sorted(f"{m['graph']}|{m['gid']}" for m in L.values())
                lg = gid("ls", *members)
                first = next(iter(L.values()))
                self.q("MERGE (l:LogicalState {gid:$g}) SET l.summary = $s, l.domain = $d, l.devices = $devs "
                       "WITH l MATCH (sr:SiteRef {gid:$sr}) MERGE (l)-[:ON_SITE]->(sr)",
                       g=lg, s=first["summary"], d=domain, devs=sorted(L), sr=gid("sr", domain))
                if first["emb"]:
                    self.q("MATCH (l:LogicalState {gid:$g}) SET l.emb = vecf32($v)", g=lg, v=first["emb"])
                for dev, m in L.items():
                    self.q("""MERGE (d:DeviceStateRef {gid:$dg}) SET d.ref_graph = $graph, d.ref_gid = $sg, d.device = $dev
                              WITH d MATCH (l:LogicalState {gid:$lg}) MERGE (l)-[r:REALISED_AS]->(d) SET r.device = $dev""",
                           dg=gid("dr2", m["graph"], m["gid"]), graph=m["graph"], sg=m["gid"], dev=dev, lg=lg)
                L["_gid"] = lg
                total += 1
                aligned += int(len([k for k in L if not k.startswith("_")]) > 1)
            self._device_deltas(sites, logical, base)
        return {"logical_states": total, "aligned": aligned}

    def _device_deltas(self, sites: list[dict], logical: list[dict], base: str) -> None:
        """How many more clicks the same screen takes on another device than on the base device."""
        keys = {s["device"]: s["key"] for s in sites}
        entries = {dev: entry_state(self.store, k) for dev, k in keys.items()}
        entry_ls = next((L["_gid"] for L in logical if base in L and entries.get(base)
                         and L[base]["gid"] == entries[base][0]), None)
        if entry_ls is None:
            return
        for L in logical:
            steps = {}
            for dev, m in L.items():
                if dev.startswith("_") or not entries.get(dev):
                    continue
                routes = best_routes(self.store, keys[dev], entries[dev][0], m["gid"], 1)
                if routes:
                    steps[dev] = len(routes[0].steps)
            if len(steps) < 2 or base not in steps:
                continue
            self.q("MATCH (l:LogicalState {gid:$g}) SET l.device_steps = $s", g=L["_gid"], s=json.dumps(steps))
            for dev, n in steps.items():
                if dev != base and n != steps[base]:
                    self.q("""MATCH (e:LogicalState {gid:$e}), (l:LogicalState {gid:$l})
                              MERGE (e)-[d:DEVICE_DELTA {from:$f, to:$t}]->(l) SET d.extra_steps = $x""",
                           e=entry_ls, l=L["_gid"], f=base, t=dev, x=n - steps[base])

    # ---------- StepPatterns and FlowPatterns ----------
    def _step_and_flow_patterns(self) -> dict:
        ls_rows = self.store.ro_query(self.key, "MATCH (l:LogicalState) RETURN l.gid, coalesce(l.summary, '') ORDER BY l.gid")
        patterns: list[dict] = []
        ls_to_sp: dict[str, str] = {}
        for lg, summary in ls_rows:
            data = complete_json(self.llm, "You label screens by their role in user flows.",
                                 f"Name the role of this screen in a user flow in 1-3 words (e.g. cart review, "
                                 f"product detail). Screen: {summary}\n" + 'Return {"role": "..."}',
                                 default={"role": " ".join(summary.split()[:3]) or "screen"})
            role = str(data.get("role") if isinstance(data, dict) else "") or "screen"
            v = self.embedder.embed([role])[0]
            best = next((p for p in patterns if cosine(v, p["emb"]) >= self.s.uipattern_sim), None)
            if best is None:
                best = {"gid": gid("sp", role.lower()), "role": role, "emb": v}
                patterns.append(best)
                self.q("MERGE (p:StepPattern {gid:$g}) SET p.role = $r, p.emb = vecf32($v)", g=best["gid"], r=role, v=v)
            self.q("MATCH (l:LogicalState {gid:$l}), (p:StepPattern {gid:$p}) MERGE (l)-[i:INSTANCE_OF]->(p) "
                   "SET i.conf = $c", l=lg, p=best["gid"], c=round(cosine(v, best["emb"]), 4))
            ls_to_sp[lg] = best["gid"]
        state_to_ls = {(g, s): lg for lg, g, s in self.store.ro_query(
            self.key, "MATCH (l:LogicalState)-[:REALISED_AS]->(d:DeviceStateRef) RETURN l.gid, d.ref_graph, d.ref_gid")}
        shapes: dict[tuple, list[dict]] = defaultdict(list)
        for site in self._sites():
            for wf, name, goal, emb in self.ro(site["key"], "MATCH (w:Workflow) RETURN w.gid, w.name, w.goal, w.emb "
                                                             "ORDER BY w.gid"):
                states = [r[0] for r in self.ro(site["key"], """MATCH (s:WStep)-[:STEP_OF]->(:Workflow {gid:$w}),
                                                                (s)-[:FROM]->(st:State) RETURN st.gid ORDER BY s.order""", w=wf)]
                seq = []
                for sg in states:
                    sp = ls_to_sp.get(state_to_ls.get((site["key"], sg)))
                    if sp and (not seq or seq[-1] != sp):
                        seq.append(sp)
                if seq:
                    shapes[tuple(seq)].append({"graph": site["key"], "gid": wf, "goal": goal or name, "domain": site["domain"],
                                               "emb": list(emb) if emb else self.embedder.embed([goal or name])[0],
                                               "len": len(states)})
        for shape, members in shapes.items():
            data = complete_json(self.llm, "You name generic user flows.",
                                 f"Name this flow generically in 2-4 words. Goals: {[m['goal'] for m in members][:6]}\n"
                                 'Return {"name": "..."}', default={"name": members[0]["goal"]})
            name = str(data.get("name") if isinstance(data, dict) else "") or members[0]["goal"]
            fg = gid("fp", *shape)
            self.q("MERGE (f:FlowPattern {gid:$g}) SET f.name = $n, f.support = $s, f.avg_len = $a, f.emb = vecf32($v)",
                   g=fg, n=name, s=len({m["domain"] for m in members}), a=sum(m["len"] for m in members) / len(members),
                   v=_mean([m["emb"] for m in members]))
            for order, sp in enumerate(shape):
                self.q("MATCH (f:FlowPattern {gid:$f}), (p:StepPattern {gid:$p}) MERGE (f)-[s:STEP {order:$o}]->(p)",
                       f=fg, p=sp, o=order)
            for a, b in zip(shape, shape[1:]):
                self.q("MATCH (a:StepPattern {gid:$a}), (b:StepPattern {gid:$b}) MERGE (a)-[n:NEXT]->(b) "
                       "SET n.count = coalesce(n.count, 0) + 1", a=a, b=b)
            for m in members:
                self.q("""MERGE (r:WorkflowRef {gid:$rg}) SET r.ref_graph = $graph, r.ref_gid = $wf
                          WITH r MATCH (f:FlowPattern {gid:$f}), (s:SiteRef {gid:$sr})
                          MERGE (r)-[:INSTANCE_OF]->(f) MERGE (r)-[:ON_SITE]->(s)""",
                       rg=gid("wr", m["graph"], m["gid"]), graph=m["graph"], wf=m["gid"], f=fg, sr=gid("sr", m["domain"]))
        return {"step_patterns": len(patterns), "flow_patterns": len(shapes)}

    def _followed_by(self) -> dict:
        """UIPattern -> UIPattern: which kinds of component tend to appear after a click on another kind."""
        comp_to_up = {(g, c): u for g, c, u in self.store.ro_query(
            self.key, "MATCH (r:ComponentRef)-[:INSTANCE_OF]->(u:UIPattern) RETURN r.ref_graph, r.ref_gid, u.gid")}
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for site in self._sites():
            rows = self.ro(site["key"], """
                MATCH (a:State)-[r:ACTION]->(b:State)
                MATCH (k:Component)-[:HAS_CSTATE]->(:ComponentState)-[:CONTAINS]->(e:Element) WHERE e.gid = r.el_gid
                MATCH (b)-[:COMPOSED_OF]->(:ComponentState)<-[:HAS_CSTATE]-(k2:Component)
                WHERE NOT (a)-[:COMPOSED_OF]->(:ComponentState)<-[:HAS_CSTATE]-(k2)
                RETURN k.gid, k2.gid, count(*)""")
            for k1, k2, n in rows:
                u1, u2 = comp_to_up.get((site["key"], k1)), comp_to_up.get((site["key"], k2))
                if u1 and u2 and u1 != u2:
                    counts[(u1, u2)] += n
        totals: dict[str, int] = defaultdict(int)
        for (a, _), n in counts.items():
            totals[a] += n
        for (a, b), n in counts.items():
            self.q("MATCH (a:UIPattern {gid:$a}), (b:UIPattern {gid:$b}) MERGE (a)-[f:USUALLY_FOLLOWED_BY]->(b) "
                   "SET f.count = $n, f.p = $p", a=a, b=b, n=n, p=n / totals[a])
        return {"followed_by": len(counts)}

    def _algorithms(self) -> None:
        if self.store.ro_query(self.key, "MATCH ()-[f:USUALLY_FOLLOWED_BY]->() RETURN count(f)")[0][0]:
            self.q("CALL algo.pageRank('UIPattern', 'USUALLY_FOLLOWED_BY') YIELD node, score SET node.pagerank = score")
        self.q("MATCH (u:UIPattern) WHERE u.pagerank IS NULL SET u.pagerank = 0.0")
        if self.store.ro_query(self.key, "MATCH (:StepPattern)-[n:NEXT]->() RETURN count(n)")[0][0]:
            self.q("CALL algo.labelPropagation({nodeLabels:['StepPattern'], relationshipTypes:['NEXT']}) "
                   "YIELD node, communityId SET node.community = communityId")
