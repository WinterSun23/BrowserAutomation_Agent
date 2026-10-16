from __future__ import annotations

from ..graph.map_writer import MapWriter
from ..ids import gid
from ..llm import complete_json
from ..normalize import normalize_text

_STATE_SYSTEM = "You describe web page states for a browser automation agent. Be concrete and short."


class Curator:
    def __init__(self, store, key, settings, embedder, llm):
        self.store, self.key, self.s, self.embedder, self.llm = store, key, settings, embedder, llm
        self.writer = MapWriter(store, key, settings)

    def q(self, cypher: str, **params):
        return self.store.query(self.key, cypher, params)

    def ro(self, cypher: str, **params):
        return self.store.ro_query(self.key, cypher, params)

    # ---------- annotation (Layer-1 semantics) ----------
    def annotate(self, limit: int = 200) -> int:
        rows = self.ro("""
            MATCH (p:Page)-[:HAS_STATE]->(s:State) WHERE s.summary IS NULL
            OPTIONAL MATCH (s)-[:COMPOSED_OF]->(cs:ComponentState)<-[:HAS_CSTATE]-(k:Component)
            OPTIONAL MATCH (cs)-[:CONTAINS]->(x:Element)
            RETURN s.gid, p.title, p.url_template, collect(DISTINCT k.kind), collect(DISTINCT x.name_sample)
            LIMIT $n""", n=limit)
        for sg, title, tpl, kinds, names in rows:
            names = [n for n in names if n][:40]
            prompt = (f"Page title: {title}\nURL pattern: {tpl}\nRegions: {', '.join(k for k in kinds if k)}\n"
                      f"Dialog open: {'yes' if 'dialog' in kinds else 'no'}\nInteractive elements: {'; '.join(names)}\n"
                      'Return {"summary": "<one sentence: what this screen is and what a user can do>", '
                      '"intents": ["<2-4 short user goals this screen fulfils, e.g. view shopping cart>"]}')
            data = complete_json(self.llm, _STATE_SYSTEM, prompt,
                                 default={"summary": f"{title} ({tpl})", "intents": [f"open {title}"]})
            if not isinstance(data, dict):
                data = {"summary": f"{title} ({tpl})", "intents": [f"open {title}"]}
            summary = str(data.get("summary") or f"{title} ({tpl})")[:300]
            intents = [str(i)[:120] for i in (data.get("intents") or [])][:4] or [f"open {title}"]
            vecs = self.embedder.embed([summary] + intents)
            self.q("MATCH (s:State {gid:$g}) SET s.summary = $sum, s.emb = vecf32($v)", g=sg, sum=summary, v=vecs[0])
            for text, v in zip(intents, vecs[1:]):
                ig = gid("in", normalize_text(text))
                self.q("""MERGE (i:Intent {gid:$ig})
                          SET i.text = coalesce(i.text, $t), i.source = coalesce(i.source, 'llm'),
                              i.hits = coalesce(i.hits, 0), i.emb = vecf32($v)
                          WITH i MATCH (s:State {gid:$g})
                          MERGE (i)-[f:FULFILLED_BY]->(s) SET f.score = coalesce(f.score, 1.0)""",
                       ig=ig, t=text, v=v, g=sg)
        self._embed_elements()
        self._name_components()
        return len(rows)

    def _embed_elements(self) -> None:
        rows = self.ro("MATCH (x:Element) WHERE x.emb IS NULL RETURN x.gid, x.role, x.name_sample LIMIT 2000")
        if rows:
            vecs = self.embedder.embed([f"{r[1]} {r[2] or ''}" for r in rows])
            self.q("UNWIND $rows AS r MATCH (x:Element {gid:r.g}) SET x.emb = vecf32(r.v)",
                   rows=[{"g": r[0], "v": v} for r, v in zip(rows, vecs)])

    def _name_components(self) -> None:
        rows = self.ro("""MATCH (k:Component) WHERE k.emb IS NULL
                          OPTIONAL MATCH (k)-[:HAS_CSTATE]->(:ComponentState)-[:CONTAINS]->(x:Element)
                          RETURN k.gid, k.kind, k.name, collect(DISTINCT x.name_sample)""")
        if not rows:
            return
        labels = [f"{kind}: {name + ' - ' if name else ''}{', '.join([n for n in names if n][:4])}"
                  for _, kind, name, names in rows]
        vecs = self.embedder.embed(labels)
        self.q("UNWIND $rows AS r MATCH (k:Component {gid:r.g}) SET k.label = r.l, k.emb = vecf32(r.v)",
               rows=[{"g": r[0], "l": l, "v": v} for r, l, v in zip(rows, labels, vecs)])

    # ---------- algorithms ----------
    def run_algorithms(self) -> None:
        if not self.ro("MATCH ()-[r:ACTION]->() RETURN count(r)")[0][0]:
            self.q("MATCH (s:State) SET s.pagerank = 0.0, s.betweenness = 0.0, s.community = 0")
            return
        self.q("CALL algo.pageRank('State', 'ACTION') YIELD node, score SET node.pagerank = score")
        self.q("CALL algo.betweenness({nodeLabels:['State'], relationshipTypes:['ACTION'], samplingSize:64, "
               "samplingSeed:42}) YIELD node, score SET node.betweenness = score")
        self.q("CALL algo.labelPropagation({nodeLabels:['State'], relationshipTypes:['ACTION'], maxIterations:20}) "
               "YIELD node, communityId SET node.community = communityId")
        self.q("MATCH (s:State) WHERE s.pagerank IS NULL SET s.pagerank = 0.0, s.betweenness = 0.0, s.community = -1")

    def build_features(self) -> int:
        rows = self.ro("MATCH (s:State) RETURN s.community, collect([s.gid, coalesce(s.pagerank, 0.0), "
                       "coalesce(s.summary, '')])")
        keep = []
        for community, members in rows:
            members.sort(key=lambda m: m[1], reverse=True)
            fg = gid("ft", members[0][0])
            summaries = "; ".join(m[2] for m in members[:8] if m[2])
            data = complete_json(self.llm, "You name areas of a website.",
                                 f"Name this area of the site in 2-4 words. Screens: {summaries}\n"
                                 'Return {"name": "..."}', default={"name": f"Area {len(keep) + 1}"})
            name = str(data.get("name") if isinstance(data, dict) else "") or f"Area {len(keep) + 1}"
            v = self.embedder.embed([f"{name}. {summaries}"])[0]
            self.q("MERGE (f:Feature {gid:$g}) SET f.community_id = $c, f.name = $n, f.size = $size, "
                   "f.medoid_gid = $m, f.summary = $sum, f.emb = vecf32($v)",
                   g=fg, c=community, n=name, size=len(members), m=members[0][0], sum=summaries[:500], v=v)
            keep.append(fg)
        self.q("MATCH (f:Feature) WHERE NOT f.gid IN $keep DELETE f", keep=keep)
        return len(keep)

    def compute_demo_scores(self) -> None:
        rows = self.ro("MATCH (s:State) RETURN s.gid, coalesce(s.pagerank, 0.0), coalesce(s.betweenness, 0.0)")
        if rows:
            mp = max(r[1] for r in rows) or 1.0
            mb = max(r[2] for r in rows) or 1.0
            self.q("UNWIND $rows AS r MATCH (s:State {gid:r.g}) SET s.demo_score = r.d",
                   rows=[{"g": g, "d": 0.6 * p / mp + 0.4 * b / mb} for g, p, b in rows])
        wf = self.ro("""MATCH (w:Workflow)<-[:STEP_OF]-(:WStep)-[:FROM]->(s:State)
                        RETURN w.gid, max(coalesce(s.pagerank, 0.0)), avg(coalesce(s.betweenness, 0.0)),
                               count(DISTINCT s.community), coalesce(w.run_count, 0), coalesce(w.success_rate, 0.0)""")
        if wf:
            mi = max(r[1] for r in wf) or 1.0
            mh = max(r[2] for r in wf) or 1.0
            mc = max(r[3] for r in wf) or 1
            mr = max(r[4] for r in wf) or 1
            self.q("UNWIND $rows AS r MATCH (w:Workflow {gid:r.g}) SET w.demo_score = r.d", rows=[
                {"g": g, "d": 0.35 * i / mi + 0.25 * h / mh + 0.2 * c / mc + 0.1 * n / mr + 0.1 * sr}
                for g, i, h, c, n, sr in wf])

    def refresh_frontiers(self, max_risk_cost: int) -> None:
        for (g,) in self.ro("MATCH (s:State) RETURN s.gid"):
            self.writer.refresh_frontier(g, max_risk_cost)

    # ---------- retention ----------
    def apply_retention(self, keep: int | None = None) -> int:
        keep = self.s.retention_runs if keep is None else keep
        runs = [r[0] for r in self.ro("MATCH (r:Run) RETURN r.gid ORDER BY r.started DESC, r.gid DESC")]
        old = runs[keep:]
        if not old:
            return 0
        protected = {r[0] for r in self.ro("MATCH (:Shot)-[:FROM_STEP]->(:Step)-[:IN_RUN]->(r:Run) "
                                           "RETURN DISTINCT r.gid")}
        last_success: dict[str, tuple[int, str]] = {}
        for wf, run, started in self.ro("MATCH (r:Run)-[:EXECUTED]->(w:Workflow) WHERE r.status = 'done' "
                                        "RETURN w.gid, r.gid, r.started"):
            if wf not in last_success or started > last_success[wf][0]:
                last_success[wf] = (started, run)
        protected |= {run for _, run in last_success.values()}
        doomed = [r for r in old if r not in protected]
        if doomed:
            self.q("MATCH (o:Observation)-[:DURING]->(:Step)-[:IN_RUN]->(r:Run) WHERE r.gid IN $g DETACH DELETE o",
                   g=doomed)
            self.q("MATCH (st:Step)-[:IN_RUN]->(r:Run) WHERE r.gid IN $g DETACH DELETE st", g=doomed)
            self.q("MATCH (r:Run) WHERE r.gid IN $g DETACH DELETE r", g=doomed)
        return len(doomed)

    def run_all(self, max_risk_cost: int) -> dict:
        annotated = self.annotate()
        self.run_algorithms()
        features = self.build_features()
        self.compute_demo_scores()
        self.refresh_frontiers(max_risk_cost)
        deleted = self.apply_retention()
        return {"annotated": annotated, "features": features, "runs_deleted": deleted}
