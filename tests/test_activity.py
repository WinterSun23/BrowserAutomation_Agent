import pytest
from sitegraph.graph.activity import ActivityLog

pytestmark = pytest.mark.integration


def test_run_steps_chain_and_observation(store, gkey):
    store.query(gkey, "CREATE (:State {gid:'st:a'}), (:State {gid:'st:b'}), (:Element {gid:'el:x'})")
    log = ActivityLog(store, gkey)
    run = log.start_run("execute", "navigator-1", request="go to b")
    s0 = log.add_step(run, 0, "click", "ac:1", "st:a", "st:b", ok=True, value_var="q")
    s1 = log.add_step(run, 1, "click", "ac:2", "st:b", "st:a", ok=False, error="timeout")
    log.add_observation("heal", {"method": "fingerprint"}, step_gid=s1, of_gid="el:x")
    log.finish_run(run, "failed")
    assert store.ro_query(gkey, "MATCH (a:Step)-[:NEXT]->(b:Step) RETURN a.gid, b.gid") == [[s0, s1]]
    assert store.ro_query(gkey, "MATCH (:Step {gid:$g})-[:FROM]->(f:State), (:Step {gid:$g})-[:TO]->(t:State) "
                                "RETURN f.gid, t.gid", {"g": s0}) == [["st:a", "st:b"]]
    assert store.ro_query(gkey, "MATCH (o:Observation)-[:DURING]->(:Step {gid:$g}), (o)-[:OF]->(e:Element) "
                                "RETURN o.kind, e.gid", {"g": s1}) == [["heal", "el:x"]]
    assert store.ro_query(gkey, "MATCH (r:Run {gid:$r}) RETURN r.status, r.kind", {"r": run}) == [["failed", "execute"]]
    assert store.ro_query(gkey, "MATCH (s:Step)-[:IN_RUN]->(:Run {gid:$r}) RETURN count(s)", {"r": run}) == [[2]]
