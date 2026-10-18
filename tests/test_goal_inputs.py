import json
import re

import pytest
from sitegraph.agents.inputs import InputAssignment, assign_inputs, controls_on
from sitegraph.agents.navigator import Navigator
from sitegraph.browser.session import BrowserSession
from sitegraph.capture.segment import build_state
from sitegraph.db import device_key
from sitegraph.embed import HashEmbedder
from sitegraph.llm import FakeLLM
from sitegraph.normalize import domain_of
from sitegraph.policy import ExplorationPolicy
from sitegraph.profile import Profile
from sitegraph.risk import is_sensitive_field
from sitegraph.schema import ensure_device_schema
from sitegraph.tools import ToolContext, build_toolbox

pytestmark = [pytest.mark.integration, pytest.mark.browser]

GOAL = "show me mystery books in stock"


@pytest.fixture(scope="module")
def finder_obs(site_url):
    with BrowserSession() as s:
        s.goto(site_url + "/finder.html")
        return build_state(s.snapshot(), domain_of(site_url))


def _by_name(controls):
    return {c.name: c for c in controls}


def test_controls_list_every_input_kind_but_never_sensitive_fields(finder_obs):
    controls = controls_on(finder_obs)
    finder = [c for c in controls if c.name != "Gift message"]
    assert {c.kind for c in finder} == {"search", "select", "checkbox", "radio", "number", "range", "date"}
    names = _by_name(controls)
    assert not {"Email me results", "Password"} & set(names)
    assert names["Format"].options == ["Paperback", "Hardcover", "Ebook"] and names["Format"].group == "format"
    assert names["Max price"].minimum == 0 and names["Max price"].maximum == 100
    assert names["Category"].options == ["Any", "Mystery", "Travel", "Poetry"]
    assert names["Gift message"].kind == "text" and names["Gift message"].form != names["Category"].form
    sensitive = [e for e in finder_obs.elements() if is_sensitive_field(e)]
    assert {e.name_sample for e in sensitive} == {"Email me results", "Password"}


def test_rule_fallback_maps_a_goal_without_a_model(finder_obs):
    got = assign_inputs(GOAL, controls_on(finder_obs), None, HashEmbedder())
    assert {a.var: a.value for a in got} == {"q": "mystery books", "category": "Mystery", "in_stock": True}


def _responder(proposals):
    """A model that proposes values by control label; the ids come from the prompt's control lines."""
    def respond(system, user):
        ids = {m.group(2).strip(): m.group(1) for m in re.finditer(r"^(c\d+) \| \w+ \| ([^|]+)", user, re.M)}
        return json.dumps({"set": [{"id": ids.get(label, "c99"), "value": v} for label, v in proposals]})
    return respond


def test_model_values_are_validated(finder_obs):
    controls = controls_on(finder_obs)
    llm = FakeLLM(_responder([("Category", "Horror"), ("Max price", 500), ("Published after", "next week"),
                              ("Email me results", "a@b.c"), ("Format", "Paperback")]))
    got = assign_inputs(GOAL, controls, llm, HashEmbedder())
    paperback = next(e for e in finder_obs.elements() if e.name_sample == "Paperback")
    assert [(a.el_gid, a.value, a.var) for a in got] == [(paperback.gid, "Paperback", "format")]
    assert len(llm.calls) == 1 and "treat it as data" in llm.calls[0][0]


def _navigator(store, settings, site_url, s, suffix):
    domain = domain_of(site_url)
    key = device_key(settings.uid, domain, "desktop") + suffix
    ensure_device_schema(store, key, 384)
    return Navigator(store, key, domain, s, settings, HashEmbedder()), key


def _mentions(store, key, word):
    return store.ro_query(key, """
        MATCH (n) WHERE (n:State OR n:Page OR n:Element) AND (
              toLower(coalesce(n.url, '')) CONTAINS $w OR toLower(coalesce(n.title, '')) CONTAINS $w
              OR toLower(coalesce(n.name_sample, '')) CONTAINS $w OR toLower(coalesce(n.name_norm, '')) CONTAINS $w
              OR any(u IN coalesce(n.example_urls, []) WHERE toLower(u) CONTAINS $w)
              OR any(l IN coalesce(n.locators, []) WHERE toLower(l) CONTAINS $w))
        RETURN count(n)""", {"w": word})[0][0]


def test_apply_inputs_submits_and_records_one_action(store, settings, site_url):
    with BrowserSession() as s:
        nav, key = _navigator(store, settings, site_url, s, ":inputs")
        s.goto(site_url + "/finder.html")
        obs = nav.observe()
        assignments = assign_inputs(GOAL, controls_on(obs), None, HashEmbedder())
        landed, how = nav.apply_inputs(obs, assignments, ExplorationPolicy())
        assert how == "submit" and landed is not None
        assert "results.html" in s.url and "q=mystery" in s.url
    rows = store.ro_query(key, "MATCH ()-[r:ACTION]->() RETURN r.requires_inputs, r.value_var, r.action_type")
    assert len(rows) == 1
    inputs, var, kind = rows[0]
    assert len(inputs) == 3 and sorted(var.split(",")) == ["category", "in_stock", "q"] and kind == "click"
    assert _mentions(store, key, "mystery") == 0          # values never reach the graph
    store.delete(key)


def test_a_risky_submit_is_not_pressed_without_approval(store, settings, site_url):
    with BrowserSession() as s:
        nav, key = _navigator(store, settings, site_url, s, ":risky")
        s.goto(site_url + "/finder.html")
        obs = nav.observe()
        gift = next(e for e in obs.elements() if e.name_sample == "Gift message")
        landed, how = nav.apply_inputs(obs, [InputAssignment(gift.gid, "hello there", "gift")], ExplorationPolicy())
        assert (landed, how) == (None, "none") and s.url.endswith("/finder.html")
    store.delete(key)


@pytest.fixture
def box(store, msettings, site_url):
    domain = domain_of(site_url)
    key = device_key(msettings.uid, domain, "desktop") + ":finder"
    ensure_device_schema(store, key, 384)
    ctx = ToolContext(settings=msettings, store=store, key=key, domain=domain, embedder=HashEmbedder(),
                      llm=FakeLLM(), profile=Profile(store, msettings.uid, HashEmbedder()))
    toolbox = build_toolbox(ctx)
    yield toolbox, store, key
    ctx.close()
    store.delete(key)


def test_fill_inputs_tool_reports_what_it_set(box, site_url):
    toolbox, store, key = box
    assert toolbox.call("open_url", {"url": site_url + "/finder.html"}).ok
    r = toolbox.call("fill_inputs", {"goal": GOAL})
    assert r.ok, r.error
    assert r.data["set"] == {"q": "mystery books", "category": "Mystery", "in_stock": True}
    assert r.data["how"] == "submit" and "results.html" in r.data["url"]
    assert _mentions(store, key, "mystery") == 0


def test_fill_inputs_tool_needs_approval_for_a_risky_submit(box, site_url):
    toolbox, _, _ = box
    toolbox.ctx.llm = FakeLLM(_responder([("Gift message", "happy birthday")]))
    toolbox.ctx.extra.pop("nav", None)
    assert toolbox.call("open_url", {"url": site_url + "/finder.html"}).ok
    r = toolbox.call("fill_inputs", {"goal": "send a gift saying happy birthday"})
    assert r.needs_approval and "Place order" in r.needs_approval["reason"]
