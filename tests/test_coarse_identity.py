from dataclasses import replace

import pytest
from sitegraph.agents.navigator import Navigator
from sitegraph.blobs import BlobStore
from sitegraph.browser.session import BrowserSession
from sitegraph.capture.segment import CONTENT_KINDS, build_state, count_bucket
from sitegraph.db import device_key
from sitegraph.embed import HashEmbedder
from sitegraph.graph.map_writer import MapWriter
from sitegraph.migrate import rehash
from sitegraph.normalize import domain_of
from sitegraph.policy import ExplorationPolicy
from sitegraph.schema import ensure_device_schema
from sitegraph.versions import ALGO_VERSION
from tests.helpers import node, page

pytestmark = [pytest.mark.integration, pytest.mark.browser]


def _snap(s, url):
    s.goto(url)
    return s.snapshot()


def test_content_changes_do_not_make_new_states(site_url):
    with BrowserSession() as s:
        one, two = _snap(s, site_url + "/feed.html?seed=1"), _snap(s, site_url + "/feed.html?seed=2")
    domain = domain_of(site_url)
    assert {n.name for n in one.nodes if n.role == "button"} != {n.name for n in two.nodes if n.role == "button"}
    assert build_state(one, domain).state_gid == build_state(two, domain).state_gid
    assert build_state(one, domain, identity="coarse").state_gid == build_state(two, domain, identity="coarse").state_gid
    assert build_state(one, domain, identity="fine").state_gid != build_state(two, domain, identity="fine").state_gid


def test_navigation_differences_still_do():
    def shop(nav_href, post):
        return page("http://s.test/p.html",
                    node(0, "navigation", "Main", gcss="nav"),
                    node(1, "link", "Section", parent=0, depth=1, href=f"http://s.test/{nav_href}.html"),
                    node(2, "main", "", gcss="main"),
                    node(3, "button", f"Like {post}", parent=2, depth=1, gcss="main > button"))
    assert shop("travel", "cats").state_gid == shop("travel", "dogs").state_gid     # content: merged
    assert shop("travel", "cats").state_gid != shop("mystery", "cats").state_gid    # navigation: kept apart


def test_an_open_dialog_or_a_form_keeps_states_apart(site_url):
    domain = domain_of(site_url)
    with BrowserSession() as s:
        s.goto(site_url + "/product.html?id=1")
        s.page.click("#product-add")                       # one item in the cart (localStorage)
        full = _snap(s, site_url + "/cart.html")
        empty = _snap(s, site_url + "/cart_empty.html")
        closed = _snap(s, site_url + "/filters.html")
        s.page.click("text=Show help")
        s.settle()
        opened = s.snapshot()
    empty.url = full.url                                   # same address, different screen
    assert build_state(full, domain).state_gid != build_state(empty, domain).state_gid
    assert build_state(closed, domain).state_gid != build_state(opened, domain).state_gid


def test_count_buckets():
    assert [count_bucket(n) for n in (1, 3, 7, 40)] == ["1", "2-4", "5-19", "20+"]
    assert {"main", "list", "feed", "grid", "table", "article", "region"} == CONTENT_KINDS


def test_rehash_moves_a_fine_graph_to_coarse(store, settings, site_url, tmp_path):
    assert ALGO_VERSION == 3
    domain = domain_of(site_url)
    key = device_key(settings.uid, domain, "desktop") + ":coarse"
    ensure_device_schema(store, key, 384)
    blobs = BlobStore(tmp_path / "blobs")
    fine = replace(settings, state_identity="fine")
    MapWriter(store, key, fine).ensure_site(domain, "desktop", "Desktop Chrome", "hash", 384)
    with BrowserSession() as s:
        Navigator(store, key, domain, s, fine, HashEmbedder(), blobs=blobs).explore(
            site_url + "/feed.html", ExplorationPolicy(max_depth=1))
    feed_states = "MATCH (:Page {url_template:'/feed.html?seed'})-[:HAS_STATE]->(s:State) RETURN count(s)"
    before_feed = store.ro_query(key, feed_states)[0][0]
    before = store.ro_query(key, "MATCH (s:State) RETURN count(s)")[0][0]
    assert before_feed >= 2                                   # fine: one state per seed
    report = rehash(store, key, blobs, replace(settings, state_identity="coarse"), domain)
    assert report["dropped_actions"] == 0
    assert store.ro_query(key, "MATCH (s:State) RETURN count(s)")[0][0] <= before
    assert store.ro_query(key, feed_states)[0][0] < before_feed  # seeds 1 and 2 are one coarse state now
    store.delete(key)
