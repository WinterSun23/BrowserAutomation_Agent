import pytest
from sitegraph.agents.curator import Curator
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
def shop(store, msettings, site_url):
    """The demo shop explored and curated once (no real LLM)."""
    domain = domain_of(site_url)
    key = device_key(msettings.uid, domain, "desktop")
    ensure_device_schema(store, key, 384)
    with BrowserSession() as s:
        Navigator(store, key, domain, s, msettings, HashEmbedder()).explore(
            site_url + "/index.html", ExplorationPolicy(max_depth=2))
    Curator(store, key, msettings, HashEmbedder(), FakeLLM(site_responder)).run_all(1)
    return store, msettings, key, domain, site_url


@pytest.fixture
def box(shop):
    store, settings, key, domain, _ = shop
    ctx = ToolContext(settings=settings, store=store, key=key, domain=domain, embedder=HashEmbedder(),
                      llm=FakeLLM(site_responder), profile=Profile(store, settings.uid, HashEmbedder()))
    toolbox = build_toolbox(ctx)
    yield toolbox
    ctx.close()


def _open(box, shop, page):
    r = box.call("open_url", {"url": f"{shop[4]}/{page}"})
    assert r.ok, r.error
    obs = box.call("observe", {})
    assert obs.ok, obs.error
    return obs.data


def _id(observed, role, name):
    return next(e["id"] for e in observed["elements"] if e["role"] == role and e["name"] == name)


def test_observe_lists_interactive_elements_with_ids_and_is_untrusted(box, shop):
    observed = _open(box, shop, "index.html")
    ids = [e["id"] for e in observed["elements"]]
    assert len(ids) == len(set(ids)) and all(i.startswith("e") for i in ids)
    pairs = {(e["role"], e["name"]) for e in observed["elements"]}
    assert ("link", "Products") in pairs and ("searchbox", "Search products") in pairs
    assert box.call("observe", {}).to_message_text().startswith("[untrusted website content")


def test_click_safe_element_navigates_and_records_the_action(box, shop):
    store, _, key, _, _ = shop
    observed = _open(box, shop, "index.html")
    before = store.ro_query(key, "MATCH ()-[r:ACTION]->() RETURN count(r)")[0][0]
    r = box.call("click", {"element_id": _id(observed, "link", "Browse all products")})
    assert r.ok and r.data["url"].endswith("/products.html") and r.data["page_changed"] is True
    assert store.ro_query(key, "MATCH ()-[r:ACTION]->() RETURN count(r)")[0][0] >= before
    stale = box.call("click", {"element_id": "e1"})                      # ids are invalid until the next observe
    assert not stale.ok and "observe" in stale.error


def test_click_checkout_needs_approval_then_runs_once(box, shop):
    observed = _open(box, shop, "cart.html")
    checkout = _id(observed, "button", "Checkout")
    page = box.ctx.browser().page
    blocked = box.call("click", {"element_id": checkout})
    assert blocked.needs_approval and "submit_mutating" in blocked.needs_approval["reason"]
    assert page.url.endswith("/cart.html")                                # nothing was clicked
    box.ctx.approvals.approve(blocked.needs_approval["id"])
    assert box.call("click", {"element_id": checkout}).ok and page.url.endswith("/checkout.html")
    observed = _open(box, shop, "cart.html")                              # same page, same ids again
    again = box.call("click", {"element_id": _id(observed, "button", "Checkout")})
    assert again.needs_approval and again.needs_approval["id"] != blocked.needs_approval["id"]   # one-time


def test_external_site_needs_approval(box, shop):
    r = box.call("open_url", {"url": "http://example.org/"})
    assert r.needs_approval and "leaves the explored site" in r.needs_approval["reason"]


def test_fill_never_persists_the_typed_value(box, shop):
    store, settings, key, _, _ = shop
    secret = "zq-unique-typed-value-991"
    observed = _open(box, shop, "index.html")
    filled = box.call("fill", {"element_id": _id(observed, "searchbox", "Search products"), "value": secret})
    assert filled.ok and secret not in filled.to_message_text() and filled.data["variable"]
    assert box.redacted("fill", {"element_id": "e3", "value": secret})["value"] == "<hidden>"
    clicked = box.call("click", {"element_id": _id(observed, "button", "Search")})
    assert clicked.ok and "results.html" in clicked.data["url"]
    assert box.call("observe", {}).ok
    stored = []
    for q in ("MATCH (n) RETURN n", "MATCH ()-[r]->() RETURN r"):
        stored += [str(x[0].properties) for x in store.ro_query(key, q)]
    assert stored and not any(secret.lower() in s.lower() for s in stored)       # URL, titles, names: scrubbed
    assert any("results.html" in s for s in stored)                               # ... but the page itself was recorded


def test_fill_refuses_passwords(box, shop):
    store, settings, key, domain, _ = shop
    _open(box, shop, "about.html")
    box.ctx.browser().page.evaluate(
        "() => document.querySelector('main').insertAdjacentHTML('beforeend', "
        "'<label for=\"p\">Password</label><input id=\"p\" type=\"password\">')")
    observed = box.call("observe", {}).data
    pw = next(e["id"] for e in observed["elements"] if e["role"] == "textbox")
    r = box.call("fill", {"element_id": pw, "value": "hunter2"})
    assert not r.ok and "password" in r.error


def test_list_items_returns_live_texts(box, shop):
    observed = _open(box, shop, "products.html")
    listed = next(e for e in observed["elements"] if e["role"] == "link" and e.get("repeated_items"))
    items = box.call("list_items", {"element_id": listed["id"]}).data["items"]
    assert "Green Cap" in items and len(items) == 6


def test_click_template_item_by_name(box, shop):
    observed = _open(box, shop, "products.html")
    listed = next(e for e in observed["elements"] if e["role"] == "link" and e.get("repeated_items"))
    r = box.call("click", {"element_id": listed["id"], "item": "Green Cap"})
    assert r.ok and "product.html?id=3" in r.data["url"]


def test_hover_and_scroll_and_read_text(box, shop):
    observed = _open(box, shop, "index.html")
    assert box.call("scroll", {"direction": "bottom"}).ok
    text = box.call("read_text", {}).data["text"]
    assert "Welcome to Demo Shop" in text
    assert box.call("read_text", {}).to_message_text().startswith("[untrusted")
    account = _id(observed, "button", "Account")
    assert box.call("hover", {"element_id": account}).ok


def test_find_screens_plan_route_and_run_route(box, shop):
    hits = box.call("find_screens", {"query": "open cart"})
    assert hits.ok and hits.data[0]["kind"] == "State"
    planned = box.call("plan_route", {"to_state": hits.data[0]["gid"]})
    assert planned.ok and planned.data["route_id"] and "reliability" in planned.data["explanation"]
    ran = box.call("run_route", {"route_id": planned.data["route_id"]})
    assert ran.ok and ran.data["status"] == "done" and ran.data["url"].endswith("/cart.html")
    assert box.call("describe_state", {"state_gid": hits.data[0]["gid"]}).data["next_actions"] is not None
    assert not box.call("run_route", {"route_id": "r99"}).ok


def test_run_goal_then_save_workflow(box, shop):
    res = box.call("run_goal", {"goal": "open cart"})
    assert res.ok and res.data["status"] == "done" and res.data["url"].endswith("/cart.html")
    saved = box.call("save_workflow", {"name": "Go to cart", "goal": "open my cart"})
    assert saved.ok and saved.data["workflow"].startswith("wf:")
    assert any(w["gid"] == saved.data["workflow"] for w in box.call("list_workflows", {}).data)
    assert {"next_actions", "frontier", "demo_candidates"} <= set(box.call("recommend", {}).data)


def test_run_goal_says_when_the_goal_needs_an_approved_action(box, shop):
    res = box.call("run_goal", {"goal": "go to checkout"})
    assert res.ok and res.data["url"].endswith("/cart.html")                  # the nearest allowed screen
    assert res.data["status"] == "stopped_short" and res.data["goal_reached"] is False
    assert res.data["blocked_by"]["element"] == "Checkout" and res.data["blocked_by"]["risk"] == "submit_mutating"
    assert "approval" in res.data["next_step"] and "not reached" in res.data["next_step"]
    assert box.ctx.browser().page.url.endswith("/cart.html")                  # nothing risky was clicked
    plain = box.call("run_goal", {"goal": "open cart"})                       # a goal that names no risky element
    assert plain.data["status"] == "done" and "blocked_by" not in plain.data


def test_save_workflow_needs_a_run_first(box):
    r = box.call("save_workflow", {"name": "x", "goal": "y"})
    assert not r.ok and "run_goal" in r.error


def test_memory_tools_refuse_secrets_and_allow_review_and_deletion(box):
    ok = box.call("remember", {"text": "prefers short gifs for single interactions"})
    assert ok.ok
    for bad in ("my password is hunter2", "email me at a.b@example.com", "card 4111111111111111"):
        refused = box.call("remember", {"text": bad})
        assert not refused.ok and "refused" in refused.error
    assert "gifs" in box.call("recall", {"query": "what format does the user like"}).data[0]["text"]
    mems = box.call("list_memories", {}).data
    assert any(m["gid"] == ok.data["memory"] for m in mems)
    assert box.call("forget", {"memory": ok.data["memory"]}).data["forgotten"] is True
    assert not any(m["gid"] == ok.data["memory"] for m in box.call("list_memories", {}).data)


def test_explore_never_uses_more_browsers_than_configured(box, shop):
    box.ctx.settings.navigators_per_site = 2
    res = box.call("explore", {"entry_url": f"{shop[4]}/about.html", "navigators": 99})
    assert res.ok and res.data["navigators"] == 2 and len(res.data["agents"]) == 2
    one = box.call("explore", {"entry_url": f"{shop[4]}/about.html", "navigators": 1})
    assert one.ok and one.data["navigators"] == 1


def test_make_demo_wraps_the_director(box, monkeypatch):
    from sitegraph.agents import demo_agent
    from sitegraph.demo.director import DemoResult
    from sitegraph.demo.spec import DemoSpec

    class StubDirector:                 # the tool's llm here cannot call tools, so the Demo agent uses the fixed pipeline
        def __init__(self, *a, **k): ...
        def run(self, text, start_url=None):
            return DemoResult("sb:1", [("gif", "abc.gif")], DemoSpec(["gif"], "goal", text))

    monkeypatch.setattr(demo_agent, "DemoDirector", StubDirector)
    r = box.call("make_demo", {"request": "gif of the menu"})
    assert r.ok and r.data["storyboard"] == "sb:1" and r.data["renders"] == [{"format": "gif", "file": "abc.gif"}]
