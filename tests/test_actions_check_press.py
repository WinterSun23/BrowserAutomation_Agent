import pytest
from sitegraph.agents.navigator import Navigator
from sitegraph.browser.session import BrowserSession
from sitegraph.db import device_key
from sitegraph.embed import HashEmbedder
from sitegraph.graph.queries import RouteStep
from sitegraph.llm import FakeLLM
from sitegraph.normalize import domain_of
from sitegraph.policy import ExplorationPolicy
from sitegraph.profile import Profile
from sitegraph.risk import press_risk
from sitegraph.schema import ensure_device_schema
from sitegraph.tools import ToolContext, build_toolbox
from tests.helpers import site_responder

pytestmark = [pytest.mark.integration, pytest.mark.browser]


@pytest.fixture(scope="module")
def explored(store, msettings, site_url):
    domain = domain_of(site_url)
    key = device_key(msettings.uid, domain, "desktop") + ":filters"
    ensure_device_schema(store, key, 384)
    with BrowserSession() as s:
        Navigator(store, key, domain, s, msettings, HashEmbedder()).explore(
            site_url + "/filters.html", ExplorationPolicy(max_depth=1))
    yield store, msettings, key, domain
    store.delete(key)


def _actions(store, key, kind):
    return store.ro_query(key, "MATCH (a:State)-[r:ACTION {action_type:$k}]->(b:State)<-[:HAS_STATE]-(p:Page) "
                               "RETURN r.gid, r.el_gid, r.value, r.value_var, r.requires_inputs, r.risk, "
                               "a.gid, b.gid, p.url_template", {"k": kind})


def test_exploring_toggles_facet_checkboxes(explored):
    store, _, key, _ = explored
    rows = _actions(store, key, "check")
    names = {store.ro_query(key, "MATCH (x:Element {gid:$g}) RETURN x.name_norm", {"g": r[1]})[0][0] for r in rows}
    assert names == {"in stock", "on sale"}           # one per checkbox outside a form; none for the order form
    assert all(r[2] == "True" and r[8] == "/filters.html?f" for r in rows)


def test_enter_after_a_search_box_without_button(explored):
    store, _, key, _ = explored
    rows = [r for r in _actions(store, key, "press") if r[2] == "Enter"]
    # one per state the search box is on (the base page and the facet-filtered ?f= pages)
    assert rows and len({r[6] for r in rows}) == len(rows)
    for _, el_gid, _, var, inputs, risk, _, _, tpl in rows:
        assert tpl == "/results.html?q" and inputs == [el_gid] and var == "find_items" and risk == "navigate"


def test_escape_closes_a_dialog_and_is_recorded(explored, site_url):
    store, settings, key, domain = explored
    rows = [r for r in _actions(store, key, "press") if r[2] == "Escape" and r[8] == "/filters.html"]
    assert rows and len({r[6] for r in rows}) == len(rows)       # one per state with the help dialog open
    action, el_gid, _, _, _, _, src, dst, _ = rows[0]
    opener = store.ro_query(key, "MATCH (a)-[r:ACTION]->(b:State {gid:$s}) WHERE r.action_type = 'click' "
                                 "RETURN r.gid, r.el_gid, a.gid LIMIT 1", {"s": src})[0]
    with BrowserSession() as s:
        nav = Navigator(store, key, domain, s, settings, HashEmbedder())
        s.goto(site_url + "/filters.html")
        run = nav._start("execute", "close help")
        steps = [RouteStep(action_gid=opener[0], label="open help", el_gid=opener[1], src=opener[2], dst=src),
                 RouteStep(action_gid=action, label="press Escape", el_gid=el_gid, src=src, dst=dst,
                           action_type="press")]
        outs = [nav.execute_step(st, run, i, {}) for i, st in enumerate(steps)]
        assert [o.ok for o in outs] == [True, True]
        assert s.page.locator("#help").is_hidden()


def test_press_risk():
    assert press_risk(None) == "navigate" and press_risk("submit_mutating") == "submit_mutating"


@pytest.fixture
def box(explored):
    store, settings, key, domain = explored
    ctx = ToolContext(settings=settings, store=store, key=key, domain=domain, embedder=HashEmbedder(),
                      llm=FakeLLM(site_responder), profile=Profile(store, settings.uid, HashEmbedder()))
    toolbox = build_toolbox(ctx)
    yield toolbox
    ctx.close()


def _observe(box, site_url):
    assert box.call("open_url", {"url": site_url + "/filters.html"}).ok
    return box.call("observe", {}).data["elements"]


def test_press_rejects_unknown_keys(box, site_url):
    _observe(box, site_url)
    r = box.call("press", {"key": "F13"})
    assert not r.ok and "Enter" in r.error and "Escape" in r.error


def test_press_enter_in_a_form_takes_the_forms_submit_risk(box, site_url):
    els = _observe(box, site_url)
    coupon = next(e["id"] for e in els if e["name"] == "Coupon code")
    r = box.call("press", {"key": "Enter", "element_id": coupon})
    assert r.needs_approval and "submit_mutating" in r.needs_approval["reason"]
    find = next(e["id"] for e in els if e["name"] == "Find items")
    assert box.call("fill", {"element_id": find, "value": "lamp"}).ok
    ok = box.call("press", {"key": "Enter", "element_id": find})
    assert ok.ok, ok.error
    assert "results.html?q=lamp" in ok.data["url"]


def test_set_checked_tool(box, site_url):
    els = _observe(box, site_url)
    stock = next(e["id"] for e in els if e["name"] == "In stock")
    assert next(e for e in els if e["id"] == stock)["checked"] is False
    r = box.call("set_checked", {"element_id": stock, "checked": True})
    assert r.ok, r.error
    assert r.data["checked"] is True and "f=stock" in r.data["url"]
    link = next(e["id"] for e in _observe(box, site_url) if e["role"] == "link")
    assert not box.call("set_checked", {"element_id": link, "checked": True}).ok
