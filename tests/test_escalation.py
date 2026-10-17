import json

import pytest
from sitegraph.agents.curator import Curator
from sitegraph.agents.escalation import NavigatorEscalation, make_escalation
from sitegraph.agents.navigator import Navigator
from sitegraph.browser.session import BrowserSession
from sitegraph.db import device_key
from sitegraph.embed import HashEmbedder
from sitegraph.graph.queries import best_routes, entry_state, find_targets
from sitegraph.llm import ChatMessage, FakeLLM, OfflineLLM, ScriptedLLM, ToolCall
from sitegraph.normalize import domain_of
from sitegraph.policy import ExplorationPolicy
from sitegraph.schema import ensure_device_schema
from tests.helpers import site_responder

pytestmark = [pytest.mark.integration, pytest.mark.browser]

WALL = """document.addEventListener('DOMContentLoaded', () => setTimeout(() => {
  const w = document.createElement('div'); w.id = 'cookie-wall';
  w.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:2147483000;display:flex;' +
                    'align-items:center;justify-content:center';
  w.innerHTML = '<div style="background:#fff;padding:24px"><p>We use cookies.</p>' +
                '<button id="accept">Accept cookies</button>__EXTRA__</div>';
  w.querySelector('#accept').onclick = () => w.remove();
  document.body.appendChild(w);
}, 0));"""
COOKIE_WALL = WALL.replace("__EXTRA__", "")
# the only way to close this wall is a destructive-sounding button (one button, so the page still looks the same)
RISKY_ONLY_WALL = COOKIE_WALL.replace("Accept cookies", "Delete my data and continue")


@pytest.fixture(scope="module")
def world(store, msettings, site_url):
    domain = domain_of(site_url)
    key = device_key(msettings.uid, domain, "desktop")
    ensure_device_schema(store, key, 384)
    with BrowserSession() as s:
        Navigator(store, key, domain, s, msettings, HashEmbedder()).explore(
            site_url + "/index.html", ExplorationPolicy(max_depth=2))
    Curator(store, key, msettings, HashEmbedder(), FakeLLM(site_responder)).run_all(1)
    return store, msettings, key, domain, site_url


@pytest.fixture(autouse=True)
def _forget_learned_steps(request):
    """The module shares one explored graph: start each test without steps learned by an earlier one."""
    if "world" in request.fixturenames:
        store, _, key, _, _ = request.getfixturevalue("world")
        store.query(key, "MATCH ()-[r:ACTION]->() WHERE r.pre_actions IS NOT NULL SET r.pre_actions = []")


def call(name, args=None, id="c"):
    return ChatMessage("assistant", tool_calls=[ToolCall(id, name, args or {})])


def say(text):
    return ChatMessage("assistant", content=text)


def button_id(name):
    """Scripted step: read the previous observe result and click the button called `name`."""
    def step(messages, tools):
        data = json.loads("\n".join(l for l in messages[-1].content.splitlines() if l.startswith("{")))["data"]
        return call("click", {"element_id": next(e["id"] for e in data["elements"] if e["name"] == name)}, "c2")
    return step


def _nav(world, session, llm=None, settings=None, escalate="auto"):
    store, base, key, domain, _ = world
    settings = settings or base
    esc = (make_escalation(settings, store, key, domain, HashEmbedder(), llm) if escalate == "auto" else escalate)
    return Navigator(store, key, domain, session, settings, HashEmbedder(), FakeLLM(site_responder), escalate=esc)


def _cart_step(world, nav):
    store, _, key, _, _ = world
    entry = entry_state(store, key)[0]
    target = find_targets(store, key, HashEmbedder().embed(["open cart"])[0])[0].gid
    return best_routes(store, key, entry, target, 1)[0].steps[0], entry


def _settings(world, **kw):
    from dataclasses import replace
    return replace(world[1], **kw)


def test_without_escalation_a_blocking_cookie_wall_fails_the_run(world):
    with BrowserSession() as s:
        s.context.add_init_script(COOKIE_WALL)
        res = _nav(world, s, escalate=None, settings=_settings(world, max_replans=0)).execute_goal("open cart")
        assert res.status == "failed" and not s.url.endswith("/cart.html")


def test_a_recovery_agent_dismisses_the_cookie_wall_and_the_step_succeeds(world):
    store, _, key, _, _ = world
    llm = ScriptedLLM([call("observe", id="c1"), button_id("Accept cookies"),
                       say("READY: I closed the cookie banner that covered the page.")])
    with BrowserSession() as s:
        s.context.add_init_script(COOKIE_WALL)
        res = _nav(world, s, llm).execute_goal("open cart")
        assert res.status == "done", res.explanation
        assert s.url.endswith("/cart.html")
    first_messages = llm.calls[0][0]
    prompt = first_messages[1].content                                       # the context the agent was given
    assert "open cart" in prompt and "click" in prompt.lower() and "intercepts pointer events" in prompt
    assert "Accept cookies" in prompt and "expected" in prompt.lower()       # what is on the page now, and the goal
    assert llm.images[0] and llm.images[0][:4] == b"\x89PNG"                 # a screenshot came with it (vision)
    assert [t["function"]["name"] for t in llm.calls[0][1]]                  # it was offered tools ...
    offered = {t["function"]["name"] for t in llm.calls[0][1]}
    assert {"observe", "click", "scroll"} <= offered and not ({"explore", "run_goal", "make_demo", "remember"} & offered)
    rec = store.ro_query(key, "MATCH (o:Observation {kind:'recovery'}) RETURN o.detail")
    assert rec and "cookie" in json.loads(rec[-1][0])["answer"].lower()      # what worked was written back


def test_recovery_cannot_take_a_risky_action_without_the_user(world):
    store, _, key, _, _ = world
    llm = ScriptedLLM([call("observe", id="c1"), button_id("Delete my data and continue")])
    with BrowserSession() as s:
        s.context.add_init_script(RISKY_ONLY_WALL)
        res = _nav(world, s, llm, _settings(world, max_replans=0)).execute_goal("open cart")
        assert res.status == "failed"                                       # (not 'no_route': the step really failed)
        assert s.page.locator("#cookie-wall").count() == 1                  # nothing was clicked: the wall is still up
    assert len(llm.calls) >= 2                                              # the recovery agent did run and tried
    recoveries = [json.loads(r[0]) for r in store.ro_query(key, "MATCH (o:Observation {kind:'recovery'}) RETURN o.detail")]
    assert any(r["status"] == "needs_approval" and "destructive" in " ".join(r["pending"]) for r in recoveries)


def test_recovery_gives_up_with_the_original_failure_when_it_cannot_help(world):
    llm = ScriptedLLM([say("FAILED: the page is covered and I cannot find a way to close it.")])
    with BrowserSession() as s:
        s.context.add_init_script(COOKIE_WALL)
        res = _nav(world, s, llm, _settings(world, max_replans=0)).execute_goal("open cart")
        assert res.status == "failed" and not s.url.endswith("/cart.html")


def test_no_llm_means_no_escalation_object():
    from sitegraph.config import Settings
    s = Settings()
    assert make_escalation(s, None, "k", "d", HashEmbedder(), OfflineLLM()) is None
    assert make_escalation(s, None, "k", "d", HashEmbedder(), FakeLLM(lambda a, b: "")) is None   # cannot call tools
    assert make_escalation(Settings(escalate_navigator=False), None, "k", "d", HashEmbedder(),
                           ScriptedLLM([])) is None
    assert isinstance(make_escalation(s, None, "k", "d", HashEmbedder(), ScriptedLLM([])), NavigatorEscalation)


SIZE_GATE = """document.addEventListener('DOMContentLoaded', () => setTimeout(() => {
  const g = document.createElement('div'); g.id = 'size-gate';
  g.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:2147483000;display:flex;' +
                    'align-items:center;justify-content:center';
  g.innerHTML = '<div style="background:#fff;padding:24px"><p>Choose your size first.</p>' +
                '<button id="size-m">Size M</button></div>';
  g.querySelector('#size-m').onclick = () => g.remove();
  document.body.appendChild(g);
}, 0));"""


def test_new_required_step_is_learned(world):
    """A page inserts a 'choose size' step in front of the cart link. The recovery agent gets past it once; what it did
    is stored on the route step, so the next run takes the step itself, without the model."""
    store, _, key, _, _ = world
    llm = ScriptedLLM([call("observe", id="c1"), button_id("Size M"), say("READY: I chose size M.")])
    with BrowserSession() as s:
        s.context.add_init_script(SIZE_GATE)
        nav = _nav(world, s, llm)
        step, _ = _cart_step(world, nav)
        before = store.ro_query(key, "MATCH ()-[r:ACTION {gid:$g}]->() RETURN r.seen_count, r.success_count",
                                {"g": step.action_gid})[0]
        res = nav.execute_goal("open cart")
        assert res.status == "done" and s.url.endswith("/cart.html"), res.explanation
    learned, seen, ok = store.ro_query(key, "MATCH ()-[r:ACTION {gid:$g}]->() RETURN r.pre_actions, r.seen_count, "
                                            "r.success_count", {"g": step.action_gid})[0]
    size_m = store.ro_query(key, "MATCH (x:Element) WHERE x.name_sample = 'Size M' RETURN x.gid")[0][0]
    assert learned == ["click|" + size_m]                                   # the new step is in the graph
    assert seen - before[0] == 2 and ok - before[1] == 1                     # the failed first try still counts

    silent = ScriptedLLM([])                                                 # would fail the run if it were called
    with BrowserSession() as s:
        s.context.add_init_script(SIZE_GATE)
        res = _nav(world, s, silent, _settings(world, max_replans=0)).execute_goal("open cart")
        assert res.status == "done" and s.url.endswith("/cart.html"), res.explanation
    assert silent.calls == []                                                # no model needed the second time
    with BrowserSession() as s:                                              # and a page without the gate is unaffected
        res = _nav(world, s, silent, _settings(world, max_replans=0)).execute_goal("open cart")
        assert res.status == "done" and s.url.endswith("/cart.html"), res.explanation


def test_a_learned_step_never_clicks_a_different_element(world):
    """'Accept cookies' (#accept) is learned; a later wall whose only button has the same id but a destructive name
    must not be clicked by the replay (it would bypass the approval)."""
    llm = ScriptedLLM([call("observe", id="c1"), button_id("Accept cookies"), say("READY: closed the banner.")])
    with BrowserSession() as s:
        s.context.add_init_script(COOKIE_WALL)
        assert _nav(world, s, llm).execute_goal("open cart").status == "done"
    with BrowserSession() as s:
        s.context.add_init_script(RISKY_ONLY_WALL)
        res = _nav(world, s, escalate=None, settings=_settings(world, max_replans=0)).execute_goal("open cart")
        assert res.status == "failed" and s.page.locator("#cookie-wall").count() == 1   # nothing was clicked


def test_the_number_of_escalations_per_run_is_limited(world):
    llm = ScriptedLLM([say("FAILED: no"), say("FAILED: still no"), say("FAILED: no way")])
    settings = _settings(world, max_escalations_per_run=1)
    with BrowserSession() as s:
        s.context.add_init_script(COOKIE_WALL)
        nav = _nav(world, s, llm, settings)
        step, entry = _cart_step(world, nav)
        s.goto(world[4] + "/index.html")
        run = nav.activity.start_run("execute", "t")
        assert not nav.execute_step(step, run, 0, {}).ok
        assert len(llm.calls) == 1                                           # the first failure escalated ...
        assert not nav.execute_step(step, run, 1, {}).ok
        assert len(llm.calls) == 1                                           # ... the second hit the limit
