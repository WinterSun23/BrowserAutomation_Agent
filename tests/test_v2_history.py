import pytest
from sitegraph.db import device_key
from sitegraph.git import GitOps
from sitegraph.schema import ensure_device_schema
from tests.helpers import node, page

pytestmark = pytest.mark.integration


def _pages():
    a = page("http://s.test/a.html", node(0, "navigation", "Main", gcss="nav"),
             node(1, "link", "B", parent=0, depth=1, href="http://s.test/b.html"))
    b = page("http://s.test/b.html", node(0, "navigation", "Main", gcss="nav"),
             node(1, "link", "A", parent=0, depth=1, href="http://s.test/a.html"))
    c = page("http://s.test/c.html", node(0, "navigation", "Main", gcss="nav"),
             node(1, "link", "C", parent=0, depth=1, href="http://s.test/c2.html"))
    return a, b, c


@pytest.fixture
def git(store, settings):
    ensure_device_schema(store, device_key(settings.uid, "s.test", "desktop"), 384)
    g = GitOps(store, settings.uid, "s.test", settings)
    g.init_main("desktop")
    return g


def _edge_seen(store, key):
    rows = store.ro_query(key, "MATCH ()-[r:ACTION]->() RETURN r.seen_count")
    return rows[0][0] if rows else None


def test_revert_restores_counters_and_removes_created(store, git):
    main = git.main_key("desktop")
    a, b, c = _pages()
    w = git.main_recorder("desktop")
    sa, sb = w.write_state(a), w.write_state(b)
    w.write_action(sa, sb, a.elements()[0].props())
    c0 = git.commit_main("desktop", w, "exec", "c0")
    w.write_action(sa, sb, a.elements()[0].props())
    w.write_state(c)
    c1 = git.commit_main("desktop", w, "exec", "c1")
    assert _edge_seen(store, main) == 2
    r = git.revert("desktop", c1)
    assert _edge_seen(store, main) == 1
    assert store.ro_query(main, "MATCH (:Page {url_template:'/c.html'}) RETURN count(*)") == [[0]]
    assert [x["id"] for x in git.log("desktop")][:3] == [r, c1, c0]
    git.revert("desktop", r)
    assert _edge_seen(store, main) == 2
    assert store.ro_query(main, "MATCH (:Page {url_template:'/c.html'})-[:HAS_STATE]->(:State) RETURN count(*)") == [[1]]


def test_checkout_and_compaction(store, git):
    a, b, c = _pages()
    w = git.main_recorder("desktop")
    sa, sb = w.write_state(a), w.write_state(b)
    w.write_action(sa, sb, a.elements()[0].props())
    c0 = git.commit_main("desktop", w, "exec", "c0")
    git.tag("desktop", "t0")
    w.write_state(c)
    c1 = git.commit_main("desktop", w, "exec", "c1")
    at0 = git.checkout("desktop", c0)
    assert store.ro_query(at0, "MATCH (p:Page) RETURN count(p)") == [[2]]
    at1 = git.checkout("desktop", c1)
    assert store.ro_query(at1, "MATCH (p:Page) RETURN count(p)") == [[3]]
    removed = git.compact("desktop", keep_commits=1)
    assert removed >= 1
    log = git.log("desktop")
    assert log[0]["compacted"] is False and log[-1]["compacted"] is True
    assert store.ro_query(git.checkout("desktop", c1), "MATCH (p:Page) RETURN count(p)") == [[3]]
    assert {t["name"] for t in git.tags("desktop")} >= {"desktop/genesis", "desktop/t0"}


def test_auto_checkpoint(store, git, settings):
    settings.checkpoint_every = 2
    a, b, _ = _pages()
    for page_obs in (a, b, a):
        w = git.main_recorder("desktop")
        w.write_state(page_obs)
        git.commit_main("desktop", w, "exec", "x")
    assert any(t["name"].startswith("desktop/auto-") for t in git.tags("desktop"))


def test_revert_of_a_state_merge_brings_the_merged_state_and_its_edge_back(store, git):
    from tests.test_v2_recording import _two_screens_of_one_page

    main = git.main_key("desktop")
    s1, s2, b = _two_screens_of_one_page()
    w = git.main_recorder("desktop")
    keep, merge, dst = w.write_state(s1), w.write_state(s2), w.write_state(b)
    w.write_action(keep, dst, s1.elements()[0].props())
    w.write_action(merge, dst, s2.elements()[0].props())
    git.commit_main("desktop", w, "exec", "two screens")
    w.merge_states(keep, merge)
    c = git.commit_main("desktop", w, "curator", "merge")
    assert store.ro_query(main, "MATCH (s:State {gid:$g}) RETURN count(s)", {"g": merge}) == [[0]]
    git.revert("desktop", c)
    assert store.ro_query(main, "MATCH (s:State {gid:$g}) RETURN count(s)", {"g": merge}) == [[1]]
    assert store.ro_query(main, "MATCH (:State {gid:$g})-[r:ACTION]->(:State {gid:$d}) RETURN r.seen_count",
                          {"g": merge, "d": dst}) == [[1]]
    assert store.ro_query(main, "MATCH (:State {gid:$g})-[r:ACTION]->() RETURN r.seen_count", {"g": keep}) == [[1]]
    assert store.ro_query(main, "MATCH (:Page)-[:HAS_STATE]->(s:State {gid:$g}) RETURN count(s)", {"g": merge}) == [[1]]
