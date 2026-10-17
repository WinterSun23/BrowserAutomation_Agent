import pytest
from sitegraph.capture.model import obs_from_dict, obs_to_dict
from sitegraph.config import Settings
from sitegraph.graph.activity import ActivityLog
from sitegraph.graph.map_writer import MapWriter
from sitegraph.graph.recording import Change, RecordingMapWriter, copy_activity, replay
from sitegraph.schema import ensure_device_schema
from tests.helpers import node, page

pytestmark = pytest.mark.integration


def _pages():
    a = page("http://s.test/a.html", node(0, "navigation", "Main", gcss="nav"),
             node(1, "link", "B", parent=0, depth=1, href="http://s.test/b.html"))
    b = page("http://s.test/b.html", node(0, "navigation", "Main", gcss="nav"),
             node(1, "link", "A", parent=0, depth=1, href="http://s.test/a.html"))
    return a, b


def test_map_assignment_smoke(store, gkey):
    store.query(gkey, "CREATE (:N {gid:'x', a:1, b:2})")
    store.query(gkey, "MATCH (n:N {gid:'x'}) SET n += $p", {"p": {"b": 3, "c": 4}})
    assert store.ro_query(gkey, "MATCH (n:N) RETURN properties(n)")[0][0] == {"gid": "x", "a": 1, "b": 3, "c": 4}
    store.query(gkey, "MATCH (n:N {gid:'x'}) SET n = $p", {"p": {"gid": "x", "a": 9}})
    assert store.ro_query(gkey, "MATCH (n:N) RETURN properties(n)")[0][0] == {"gid": "x", "a": 9}


def test_obs_roundtrip():
    a, _ = _pages()
    assert obs_from_dict(obs_to_dict(a)).state_gid == a.state_gid
    assert any(not hasattr(e, "hoverable") for e in obs_from_dict(obs_to_dict(a)).elements()) is False


def test_record_and_replay_reproduces_graph(store, gkey):
    for k in (gkey, gkey + "-b"):
        ensure_device_schema(store, k, 384)
    rec = RecordingMapWriter(store, gkey, Settings())
    a, b = _pages()
    sa, sb = rec.write_state(a), rec.write_state(b)
    g = rec.write_action(sa, sb, a.elements()[0].props())
    rec.write_action(sa, sb, a.elements()[0].props())
    assert rec.quarantine("ACTION", g, "test")
    assert [c.op for c in rec.changes] == ["write_state", "write_state", "write_action", "write_action", "quarantine"]
    target = MapWriter(store, gkey + "-b", Settings())
    for ch in rec.changes:
        replay(target, Change.from_props(ch.to_props()))
    q = "MATCH ()-[r:BROKEN_ACTION]->() RETURN r.gid, r.seen_count, r.reason"
    assert store.ro_query(gkey + "-b", q) == store.ro_query(gkey, q) == [[g, 2, "test"]]
    assert rec.changes[2].touched[0]["prev"] is None and rec.changes[3].touched[0]["prev"]["props"]["seen_count"] == 1
    assert target.unquarantine("ACTION", g)
    assert store.ro_query(gkey + "-b", "MATCH ()-[r:ACTION]->() RETURN count(r)") == [[1]]


def _two_screens_of_one_page():
    """Two states of the same page with different content, both linking to page b."""
    nav = node(0, "navigation", "Main", gcss="nav")
    s1 = page("http://s.test/a.html", nav, node(1, "link", "Alpha shoes", parent=0, depth=1, href="http://s.test/b.html"),
              node(2, "heading", "Shoes catalogue and sizes"))
    s2 = page("http://s.test/a.html", nav, node(1, "link", "Zebra socks", parent=0, depth=1, href="http://s.test/b.html"),
              node(2, "heading", "Completely other wording here today"),
              node(3, "complementary", "Sidebar", gcss="aside"),
              node(4, "link", "Sale items sale", parent=3, depth=1, href="http://s.test/b.html"),
              node(5, "contentinfo", "Footer", gcss="foot"))
    b = page("http://s.test/b.html", node(0, "main", "Other", gcss="m"))
    return s1, s2, b


def test_merge_and_rename_are_recorded_and_replayed(store, gkey):
    for k in (gkey, gkey + "-b"):
        ensure_device_schema(store, k, 384)
    rec = RecordingMapWriter(store, gkey, Settings())
    s1, s2, b = _two_screens_of_one_page()
    keep, merge, dst = rec.write_state(s1), rec.write_state(s2), rec.write_state(b)
    assert keep != merge
    rec.write_action(keep, dst, s1.elements()[0].props())
    rec.write_action(merge, dst, s2.elements()[0].props())
    out = rec.merge_states(keep, merge)
    assert out["actions_moved"] == 1 and merge in out["merged_state"].get("aliases", []) + [merge]
    assert store.ro_query(gkey, "MATCH (s:State {gid:$g}) RETURN count(s)", {"g": merge}) == [[0]]
    # both screens had the same link (same element identity): the two edges become one with the counters summed
    assert store.ro_query(gkey, "MATCH (:State {gid:$g})-[r:ACTION]->() RETURN count(r), sum(r.seen_count)",
                          {"g": keep}) == [[1, 2]]
    assert merge in store.ro_query(gkey, "MATCH (s:State {gid:$g}) RETURN s.aliases", {"g": keep})[0][0]
    ch = rec.changes[-1]
    assert ch.op == "merge_states" and ch.touched[1]["prev"] is not None and ch.touched[1]["prev"]["_rels"]
    rec.rename_state(keep, "The shoe catalogue", ["Shoes"], [0.1] * 384, [("open shoes", [0.2] * 384)])
    assert store.ro_query(gkey, "MATCH (s:State {gid:$g}) RETURN s.summary", {"g": keep}) == [["The shoe catalogue"]]
    assert store.ro_query(gkey, "MATCH (:Intent {text:'open shoes'})-[:FULFILLED_BY]->(s:State {gid:$g}) RETURN count(s)",
                          {"g": keep}) == [[1]]
    target = MapWriter(store, gkey + "-b", Settings())
    for c in rec.changes:
        replay(target, Change.from_props(c.to_props()))
    for q in ("MATCH (s:State) RETURN s.gid, s.summary, s.seen_count ORDER BY s.gid",
              "MATCH ()-[r:ACTION]->() RETURN r.gid, r.seen_count ORDER BY r.gid"):
        assert store.ro_query(gkey + "-b", q) == store.ro_query(gkey, q)


def test_merge_refuses_states_of_different_pages(store, gkey):
    ensure_device_schema(store, gkey, 384)
    w = MapWriter(store, gkey, Settings())
    a, b = _pages()
    sa, sb = w.write_state(a), w.write_state(b)
    with pytest.raises(ValueError, match="different screens"):
        w.merge_states(sa, sb)


def test_delete_node_keeps_structural_relationships_for_revert(store, gkey):
    ensure_device_schema(store, gkey, 384)
    rec = RecordingMapWriter(store, gkey, Settings())
    a, b = _pages()
    sa, sb = rec.write_state(a), rec.write_state(b)
    rec.delete_node("State", sb)
    prev = rec.changes[-1].touched[0]["prev"]
    assert any(r[1] == "HAS_STATE" for r in prev["_rels"])
    rec.restore_node("State", sb, {k: v for k, v in prev.items() if k != "_rels"})
    for direction, rtype, olabel, ogid, props in prev["_rels"]:
        rec.restore_rel("State", sb, direction, rtype, olabel, ogid, props)
    assert store.ro_query(gkey, "MATCH (:Page)-[:HAS_STATE]->(s:State {gid:$g}) RETURN count(s)", {"g": sb}) == [[1]]


def test_copy_activity(store, gkey):
    ensure_device_schema(store, gkey, 384)
    ensure_device_schema(store, gkey + "-d", 384)
    store.query(gkey, "CREATE (:State {gid:'st:a'})")
    store.query(gkey + "-d", "CREATE (:State {gid:'st:a'})")
    log = ActivityLog(store, gkey)
    r = log.start_run("explore", "n")
    s0 = log.add_step(r, 0, "click", None, "st:a", "st:a", ok=True)
    log.add_step(r, 1, "click", None, "st:a", None, ok=True)
    log.add_observation("heal", {}, step_gid=s0)
    assert copy_activity(store, gkey, gkey + "-d") == 1
    assert copy_activity(store, gkey, gkey + "-d") == 0
    d = gkey + "-d"
    assert store.ro_query(d, "MATCH (:Step)-[:NEXT]->(:Step) RETURN count(*)") == [[1]]
    assert store.ro_query(d, "MATCH (:Observation)-[:DURING]->(:Step)-[:FROM]->(:State) RETURN count(*)") == [[1]]


def test_factory_passes_writer_and_secrets_through():
    from sitegraph.agents.factory import make_navigator

    w = MapWriter(None, "k", Settings())
    nav = make_navigator(None, "k", "x.test", None, Settings(), None, writer=w, secrets=lambda v: "s")
    assert nav.writer is w and nav.secrets("v") == "s"
    assert make_navigator(None, "k", "x.test", None, Settings(), None).writer.key == "k"
