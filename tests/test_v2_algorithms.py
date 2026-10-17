import pytest
from sitegraph.agents.curator import Curator
from sitegraph.config import Settings
from sitegraph.embed import HashEmbedder
from sitegraph.graph.queries import frontier_states
from sitegraph.llm import FakeLLM

pytestmark = pytest.mark.integration


def test_harmonic_deadend_reachability(store, gkey):
    store.query(gkey, """CREATE (a:State:Entry {gid:'a'}), (b:State:Frontier {gid:'b', untried_count:2}),
        (c:State:Frontier {gid:'c', untried_count:2}), (z:State {gid:'z'}), (y:State {gid:'y'}),
        (a)-[:ACTION {gid:'1', cost:1.0, risk_cost:0}]->(b), (b)-[:ACTION {gid:'2', cost:1.0, risk_cost:0}]->(c),
        (b)-[:ACTION {gid:'3', cost:1.0, risk_cost:0}]->(a), (z)-[:ACTION {gid:'4', cost:1.0, risk_cost:0}]->(y)""")
    Curator(store, gkey, Settings(), HashEmbedder(), FakeLLM()).run_algorithms()
    assert store.ro_query(gkey, "MATCH (s:State) WHERE s.harmonic IS NULL RETURN count(s)") == [[0]]
    assert {r[0] for r in store.ro_query(gkey, "MATCH (s:State:DeadEnd) RETURN s.gid")} == {"c", "y"}
    assert {r[0] for r in store.ro_query(gkey, "MATCH (s:State) WHERE s.reachable = false RETURN s.gid")} == {"z", "y"}
    f = frontier_states(store, gkey)
    assert [x["gid"] for x in f] == ["b", "c"] and "harmonic" in f[0]
