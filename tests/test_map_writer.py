import math
import pytest
from sitegraph.config import Settings
from sitegraph.graph.map_writer import MapWriter
from sitegraph.schema import ensure_device_schema
from tests.helpers import node, page

pytestmark = pytest.mark.integration


def _nav(*links):
    nodes = [node(0, "navigation", "Main", gcss="nav")]
    for i, (name, href) in enumerate(links, start=1):
        nodes.append(node(i, "link", name, parent=0, depth=1, href=href))
    return nodes


@pytest.fixture
def w(store, gkey):
    ensure_device_schema(store, gkey, 384)
    return MapWriter(store, gkey, Settings())


def test_write_state_is_idempotent(w, store, gkey):
    obs = page("http://s.test/a.html", *_nav(("Home", "http://s.test/index.html")))
    assert w.write_state(obs) == w.write_state(obs) == obs.state_gid
    assert store.ro_query(gkey, "MATCH (s:State) RETURN count(s), max(s.seen_count)") == [[1, 2]]
    assert store.ro_query(gkey, "MATCH (:Page)-[:HAS_STATE]->(:State)-[:COMPOSED_OF]->(:ComponentState)"
                                "-[:CONTAINS]->(e:Element) RETURN e.role") == [["link"]]


def test_near_duplicate_resolves_to_canonical(w, store, gkey):
    links = [(f"L{i}", f"http://s.test/p{i}.html") for i in range(12)]
    a = page("http://s.test/a.html", *_nav(*links))
    b = page("http://s.test/a.html", *_nav(*links, ("Extra", "http://s.test/extra.html")))
    assert a.state_gid != b.state_gid
    ga = w.write_state(a)
    assert w.write_state(b) == ga
    assert store.ro_query(gkey, "MATCH (s:State {gid:$g}) RETURN s.aliases", {"g": ga}) == [[[b.state_gid]]]


def test_action_counters_reliability_and_cost(w, store, gkey):
    a = page("http://s.test/a.html", *_nav(("B", "http://s.test/b.html")))
    b = page("http://s.test/b.html", *_nav(("A", "http://s.test/a.html")))
    sa, sb = w.write_state(a), w.write_state(b)
    el = a.elements()[0].props()
    g1 = w.write_action(sa, sb, el)
    g2 = w.write_action(sa, sb, el)
    assert g1 == g2
    rel, cost = store.ro_query(gkey, "MATCH ()-[r:ACTION {gid:$g}]->() RETURN r.reliability, r.cost", {"g": g1})[0]
    assert math.isclose(rel, 0.75) and math.isclose(cost, 1 + 2 * -math.log(0.75), rel_tol=1e-6)
    w.record_attempt(g1, success=False)
    assert store.ro_query(gkey, "MATCH ()-[r:ACTION {gid:$g}]->() RETURN r.seen_count, r.success_count",
                          {"g": g1}) == [[3, 2]]


def test_transition_and_frontier(w, store, gkey):
    obs = page("http://s.test/a.html", node(0, "banner", gcss="header"), node(1, "button", "Account", parent=0, depth=1))
    s = w.write_state(obs)
    el = obs.elements()[0].props()
    t = w.write_transition("cp:menu", "cs:absent", "cs:open", el)
    rows = store.ro_query(gkey, "MATCH (k:Component {gid:'cp:menu'})-[:HAS_CSTATE]->(a)-[t:TRANSITION]->(b) "
                                "RETURN a.name, b.name, t.gid")
    assert rows == [["absent", "open", t]]
    assert w.refresh_frontier(s, max_risk_cost=1) == 1
    assert store.ro_query(gkey, "MATCH (s:State:Frontier) RETURN count(s)") == [[1]]
    w.mark_tried(el["gid"])
    assert w.refresh_frontier(s, max_risk_cost=1) == 0
    assert store.ro_query(gkey, "MATCH (s:State:Frontier) RETURN count(s)") == [[0]]


def test_volatile_flag_when_sample_changes(w, store, gkey):
    a = page("http://s.test/a.html", *_nav(("Cart (0)", "http://s.test/cart.html")))
    b = page("http://s.test/a.html", *_nav(("Cart (2)", "http://s.test/cart.html")))
    w.write_state(a)
    w.write_state(b)
    assert store.ro_query(gkey, "MATCH (e:Element) RETURN e.volatile, e.name_sample") == [[True, "Cart (2)"]]
