import pytest
from sitegraph.embed import HashEmbedder
from sitegraph.graph import queries as Q
from sitegraph.schema import ensure_device_schema

pytestmark = pytest.mark.integration


@pytest.fixture
def g(store, gkey):
    ensure_device_schema(store, gkey, 384)
    store.query(gkey, """
      CREATE (p:Page {gid:'pg:1', url_template:'/a', example_urls:['http://s.test/a']}),
             (a:State:Entry {gid:'st:a', summary:'home'}), (b:State {gid:'st:b'}), (c:State {gid:'st:c'}),
             (d:State:Frontier {gid:'st:d', summary:'cart page', untried_count:2}),
             (p)-[:HAS_STATE]->(a),
             (a)-[:ACTION {gid:'ac:ab', label:'click B', el_gid:'el:b', reliability:0.9, cost:1.2, risk_cost:0, action_type:'click', seen_count:9, pre_cstates:[], requires_inputs:[]}]->(b),
             (b)-[:ACTION {gid:'ac:bd', label:'click D', el_gid:'el:d', reliability:0.9, cost:1.2, risk_cost:0, action_type:'click', seen_count:9, pre_cstates:[], requires_inputs:[]}]->(d),
             (a)-[:ACTION {gid:'ac:ac', label:'buy', el_gid:'el:c', reliability:0.99, cost:1.0, risk_cost:100, action_type:'click', seen_count:2, pre_cstates:[], requires_inputs:[]}]->(c),
             (c)-[:ACTION {gid:'ac:cd', label:'go D', el_gid:'el:d2', reliability:0.99, cost:1.0, risk_cost:0, action_type:'click', seen_count:2, pre_cstates:[], requires_inputs:[]}]->(d),
             (:Element {gid:'el:b', role:'link', name_sample:'B', locators:['css=#b'], fingerprint:'{}'})""")
    v = HashEmbedder().embed(["open cart"])[0]
    store.query(gkey, "CREATE (i:Intent {gid:'in:1', text:'open cart', emb: vecf32($v)}) WITH i "
                      "MATCH (d:State {gid:'st:d'}) CREATE (i)-[:FULFILLED_BY {score:1.0}]->(d)", {"v": v})
    return gkey


def test_find_targets_by_meaning(store, g):
    t = Q.find_targets(store, g, HashEmbedder().embed(["open the cart"])[0])
    assert t and t[0].gid == "st:d" and t[0].kind == "State" and t[0].score > 0.5


def test_best_route_respects_risk(store, g):
    safe = Q.best_routes(store, g, "st:a", "st:d", max_risk_cost=1)
    assert [s.action_gid for s in safe[0].steps] == ["ac:ab", "ac:bd"]
    assert abs(safe[0].probability - 0.81) < 1e-6
    risky = Q.best_routes(store, g, "st:a", "st:d", max_risk_cost=1000)
    assert [s.action_gid for s in risky[0].steps] == ["ac:ac", "ac:cd"]
    assert "click B" in Q.explain(safe[0])


def test_same_state_route_is_empty(store, g):
    r = Q.best_routes(store, g, "st:a", "st:a", 1)
    assert r and r[0].steps == []


def test_next_actions_and_frontier_and_entry(store, g):
    acts = Q.next_actions(store, g, "st:a", max_risk_cost=1)
    assert [a["action_gid"] for a in acts] == ["ac:ab"]
    assert [f["gid"] for f in Q.frontier_states(store, g)] == ["st:d"]
    assert Q.entry_state(store, g) == ("st:a", "http://s.test/a")
    assert Q.load_element(store, g, "el:b")["locators"] == ["css=#b"]


def test_mine_sequences(store, g):
    store.query(g, """CREATE (r:Run {gid:'run:1'}),
      (s1:Step {gid:'s1', ok:true, action_gid:'ac:ab'})-[:IN_RUN]->(r), (s2:Step {gid:'s2', ok:true, action_gid:'ac:bd'})-[:IN_RUN]->(r),
      (s3:Step {gid:'s3', ok:true, action_gid:'ac:ab'})-[:IN_RUN]->(r), (s4:Step {gid:'s4', ok:true, action_gid:'ac:bd'})-[:IN_RUN]->(r),
      (s1)-[:NEXT]->(s2), (s2)-[:NEXT]->(s3), (s3)-[:NEXT]->(s4)""")
    seqs = Q.mine_sequences(store, g, min_freq=2)
    assert seqs[0]["actions"] == ["ac:ab", "ac:bd"] and seqs[0]["freq"] == 2
