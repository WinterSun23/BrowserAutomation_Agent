import pytest
from sitegraph.agents.navigator import Navigator
from sitegraph.browser.session import BrowserSession
from sitegraph.capture.segment import build_state
from sitegraph.db import device_key
from sitegraph.embed import HashEmbedder
from sitegraph.normalize import domain_of
from sitegraph.ops import Ops
from sitegraph.policy import ExplorationPolicy
from sitegraph.schema import ensure_device_schema

pytestmark = [pytest.mark.integration, pytest.mark.browser]


def _setup(store, settings, site_url, suffix=""):
    domain = domain_of(site_url)
    key = device_key(settings.uid, domain, "desktop") + suffix
    ensure_device_schema(store, key, 384)
    return domain, key


def test_observe_autoscrolls_lazy_list_without_changing_state_identity(store, settings, site_url):
    domain, key = _setup(store, settings, site_url)
    with BrowserSession() as s:
        nav = Navigator(store, key, domain, s, settings, HashEmbedder())
        s.goto(site_url + "/lazy.html")
        before = build_state(s.snapshot(), domain)            # what a non-scrolling observer sees
        after = nav.observe()                                 # what the Navigator sees
        count = lambda o: next(e.count for e in o.elements() if e.is_template)
        assert count(before) == 6 and count(after) == 24      # 4 lazy batches of 6
        assert before.state_gid == after.state_gid            # list size is not part of state identity
        assert s.page.evaluate("window.scrollY") == 0         # scrolled back to the top


def test_short_pages_are_not_scrolled(site_url):
    with BrowserSession() as s:
        s.goto(site_url + "/index.html")
        assert s.load_all(8) == 0


def test_hover_menu_is_discovered_shared_as_items_and_replayable(store, settings, site_url):
    domain, key = _setup(store, settings, site_url)
    with BrowserSession() as s:
        nav = Navigator(store, key, domain, s, settings, HashEmbedder())
        nav.explore(site_url + "/hover.html", ExplorationPolicy(max_depth=1), scope="sc-hover")
    q = lambda c, **p: store.ro_query(key, c, p)
    assert q("MATCH ()-[t:TRANSITION {action_type:'hover'}]->() RETURN count(t)")[0][0] >= 1
    assert q("MATCH (e:Element {role:'menuitem'}) RETURN count(e)")[0][0] >= 2
    assert q("MATCH ()-[r:ACTION]->() WHERE size(r.pre_cstates) > 0 RETURN count(r)")[0][0] >= 1
    ops = Ops(store, settings.uid, domain)
    assert ops.q("MATCH (w:WorkItem {scope:'sc-hover', kind:'hover'}) RETURN count(w)")[0][0] >= 1
    # the Executor can reproduce the hover: entering the 'menu open' component state hovers its trigger
    cs = q("MATCH ()-[t:TRANSITION {action_type:'hover'}]->(b) RETURN b.gid LIMIT 1")[0][0]
    with BrowserSession() as s:
        nav = Navigator(store, key, domain, s, settings, HashEmbedder())
        s.goto(site_url + "/hover.html")
        assert nav._enter_cstate(cs) is True
        assert s.page.get_by_role("menuitem", name="Sub About").is_visible()


@pytest.mark.parametrize("max_path_len, expect_path_items", [(0, False), (2, True)])
def test_max_path_len_limits_chained_steps(store, settings, site_url, max_path_len, expect_path_items):
    settings.max_path_len = max_path_len
    domain, key = _setup(store, settings, site_url)
    with BrowserSession() as s:
        nav = Navigator(store, key, domain, s, settings, HashEmbedder())
        nav.explore(site_url + "/hover.html", ExplorationPolicy(max_depth=0), scope="sc-len")
    ops = Ops(store, settings.uid, domain)
    chained = ops.q("MATCH (w:WorkItem {scope:'sc-len'}) WHERE size(w.pre) > 0 RETURN count(w)")[0][0]
    assert (chained > 0) is expect_path_items


def test_phone_sessions_skip_hover(store, settings, site_url):
    domain = domain_of(site_url)
    key = device_key(settings.uid, domain, "phone")
    ensure_device_schema(store, key, 384)
    with BrowserSession(device="phone") as s:
        nav = Navigator(store, key, domain, s, settings, HashEmbedder())
        nav.explore(site_url + "/hover.html", ExplorationPolicy(max_depth=0), scope="sc-phone")
    ops = Ops(store, settings.uid, domain)
    assert ops.q("MATCH (w:WorkItem {scope:'sc-phone', kind:'hover'}) RETURN count(w)")[0][0] == 0
    assert ops.q("MATCH (w:WorkItem {scope:'sc-phone'}) RETURN count(w)")[0][0] >= 1
