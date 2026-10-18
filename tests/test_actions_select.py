import pytest
from sitegraph.agents.navigator import Navigator
from sitegraph.browser.session import BrowserSession
from sitegraph.db import device_key
from sitegraph.embed import HashEmbedder
from sitegraph.llm import FakeLLM
from sitegraph.normalize import domain_of
from sitegraph.policy import ExplorationPolicy
from sitegraph.profile import Profile
from sitegraph.schema import ensure_device_schema
from sitegraph.tools import ToolContext, build_toolbox
from tests.helpers import site_responder

pytestmark = [pytest.mark.integration, pytest.mark.browser]


@pytest.fixture(scope="module")
def explored(store, msettings, site_url):
    domain = domain_of(site_url)
    key = device_key(msettings.uid, domain, "desktop") + ":select"
    ensure_device_schema(store, key, 384)
    with BrowserSession() as s:
        Navigator(store, key, domain, s, msettings, HashEmbedder()).explore(
            site_url + "/select.html", ExplorationPolicy(max_depth=1))
    yield store, msettings, key, domain
    store.delete(key)


def test_select_options_are_captured(site_url):
    with BrowserSession() as s:
        s.goto(site_url + "/select.html")
        snap = s.snapshot()
    sort = next(n for n in snap.nodes if n.tag == "select" and n.attrs.get("name") == "sort")
    assert sort.options[:3] == ["Relevance", "Price", "Rating"] and sort.selected_index == 0
    from sitegraph.capture.segment import build_state
    el = next(e for e in build_state(snap, domain_of(site_url)).elements() if e.name_norm == "sort by")
    assert el.options == ["Relevance", "Price", "Rating"] and el.risk == "input"


def test_exploring_tries_a_select_outside_forms_only(explored):
    store, _, key, _ = explored
    rows = store.ro_query(key, "MATCH (a:State)-[r:ACTION {action_type:'select'}]->(b:State)<-[:HAS_STATE]-(p:Page) "
                               "RETURN r.value, r.value_var, p.url_template")
    assert rows == [["Price", "sort", "/results.html?sort"]]
    # the in-form select (category) was never tried: forms are off by default
    tried = store.ro_query(key, "MATCH (x:Element {name_norm:'category'}) RETURN x.tried")
    assert tried and tried[0][0] is not True


def test_a_select_route_replays(explored, site_url):
    store, settings, key, domain = explored
    target = store.ro_query(key, "MATCH (:Page {url_template:'/results.html?sort'})-[:HAS_STATE]->(s:State) "
                                 "RETURN s.gid LIMIT 1")[0][0]
    with BrowserSession() as s:
        nav = Navigator(store, key, domain, s, settings, HashEmbedder())
        s.goto(site_url + "/select.html")
        status, cur, route, _, _ = nav.drive_to(target, nav._start("execute", "sort"), [0], {}, ExplorationPolicy())
        assert status == "done" and cur == target
        assert [st.action_type for st in route.steps] == ["select"]
        assert "results.html?sort=price" in s.url


@pytest.fixture
def box(explored):
    store, settings, key, domain = explored
    ctx = ToolContext(settings=settings, store=store, key=key, domain=domain, embedder=HashEmbedder(),
                      llm=FakeLLM(site_responder), profile=Profile(store, settings.uid, HashEmbedder()))
    toolbox = build_toolbox(ctx)
    yield toolbox
    ctx.close()


def test_select_option_tool(box, site_url):
    assert box.call("open_url", {"url": site_url + "/select.html"}).ok
    obs = box.call("observe", {}).data
    sid = next(e["id"] for e in obs["elements"] if e["role"] == "combobox" and e["name"] == "Sort by")
    assert next(e for e in obs["elements"] if e["id"] == sid)["options"] == ["Relevance", "Price", "Rating"]
    bad = box.call("select_option", {"element_id": sid, "option": "Colour"})
    assert not bad.ok and "Relevance" in bad.error and "Rating" in bad.error
    r = box.call("select_option", {"element_id": sid, "option": "Price"})
    assert r.ok, r.error
    assert r.data["page_changed"] is True and "sort=price" in r.data["url"]
