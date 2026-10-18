import json

import pytest
from sitegraph.agents.curator import Curator
from sitegraph.agents.execution_agent import ExecutionAgent
from sitegraph.agents.navigator import Navigator
from sitegraph.agents.orchestrator import Orchestrator
from sitegraph.blobs import BlobStore
from sitegraph.browser.session import BrowserSession
from sitegraph.db import device_key
from sitegraph.embed import HashEmbedder
from sitegraph.events import NullEvents
from sitegraph.llm import FakeLLM, ScriptedLLM, ChatMessage
from sitegraph.normalize import domain_of
from sitegraph.policy import ExplorationPolicy
from sitegraph.schema import ensure_device_schema
from tests.helpers import site_responder

pytestmark = [pytest.mark.integration, pytest.mark.browser]

CART_OK = {"ok": True, "evidence": "Your cart", "reason": "the cart page"}


def _llm(steps, verdict=None, answer="nothing"):
    def r(system, user):
        if "Split the request" in user:
            return json.dumps({"steps": steps})
        if "Does this page" in user:
            return json.dumps(verdict if verdict is not None else CART_OK)
        if "You answer questions about the current web page" in system:
            return json.dumps({"answer": answer})
        return site_responder(system, user)
    return FakeLLM(r)


@pytest.fixture(scope="module")
def shop(store, msettings, site_url):
    domain = domain_of(site_url)
    key = device_key(msettings.uid, domain, "desktop")
    ensure_device_schema(store, key, 384)
    with BrowserSession() as s:
        Navigator(store, key, domain, s, msettings, HashEmbedder()).explore(
            site_url + "/index.html", ExplorationPolicy(max_depth=2))
    Curator(store, key, msettings, HashEmbedder(), FakeLLM(site_responder)).run_all(1)
    msettings.goal_explore = False                     # these tests are about the request path, not goal search
    return store, msettings, site_url


def _orch(shop, llm):
    store, settings, url = shop
    o = Orchestrator(settings, store, HashEmbedder(), llm, BlobStore(settings.blob_dir), NullEvents())
    return o, o.register_site(url + "/index.html")


CART = [{"kind": "automate", "goal": "open cart"}]


def test_handle_plans_executes_and_verifies(shop):
    o, site = _orch(shop, _llm(CART))
    rr = o.handle("take me to my cart", site)
    assert rr.status == "done" and rr.plan.source == "model" and len(rr.steps) == 1
    assert rr.steps[0]["status"] == "done" and rr.steps[0]["verified"]["ok"] is True
    assert rr.steps[0]["verified"]["how"] == "model" and o.last_page["url"].endswith("/cart.html")
    flat = rr.flat()
    assert flat["status"] == "done" and flat["plan"] == [{"kind": "automate", "goal": "open cart", "params": {},
                                                          "item": None}]


def test_run_and_chat_take_the_same_path(shop):
    o, site = _orch(shop, _llm(CART))
    loop_llm = ScriptedLLM([])                          # would fail the run if the tool loop were started
    agent = ExecutionAgent(o.s, o.store, o.embedder, o.llm, None, site, o.profile, chat_llm=loop_llm, orchestrator=o)
    try:
        run = agent.ask("take me to my cart")
        assert run.status == "done" and not loop_llm.calls
        assert agent.last_result.plan.steps[0].goal == "open cart" and o.last_page["url"].endswith("/cart.html")
    finally:
        agent.close()


def test_chat_falls_back_to_the_tool_loop_only_on_failure(shop):
    o, site = _orch(shop, _llm([{"kind": "automate", "goal": "zzqx vvkk"}]))
    seen = []
    loop_llm = ScriptedLLM([lambda m, t: seen.append(m[-1].content) or ChatMessage("assistant", content="sorry")])
    agent = ExecutionAgent(o.s, o.store, o.embedder, o.llm, None, site, o.profile, chat_llm=loop_llm, orchestrator=o)
    try:
        run = agent.ask("zzqx vvkk please")
        assert run.answer == "sorry" and len(loop_llm.calls) == 1
        assert "zzqx vvkk" in seen[0] and "failed" in seen[0] and "no_target" in seen[0]
    finally:
        agent.close()


def test_a_chat_message_is_answered_without_touching_the_site(shop):
    o, site = _orch(shop, _llm([{"kind": "chat", "goal": "greeting"}]))
    loop_llm = ScriptedLLM([ChatMessage("assistant", content="hello!")])
    agent = ExecutionAgent(o.s, o.store, o.embedder, o.llm, None, site, o.profile, chat_llm=loop_llm, orchestrator=o)
    try:
        assert agent.ask("hi there").answer == "hello!"
        assert agent.last_result.steps == [] and o.last_page is None
    finally:
        agent.close()


def test_a_partial_plan_reports_what_was_done(shop):
    o, site = _orch(shop, _llm(CART + [{"kind": "automate", "goal": "zzqx vvkk"}]))
    rr = o.handle("open the cart and then zzqx", site)
    assert rr.status == "partial" and rr.failed_step == 1 and len(rr.steps) == 2, rr.steps
    assert rr.steps[0]["status"] == "done" and rr.steps[1]["status"] == "no_target"


def test_an_ask_after_a_navigation_is_answered_on_that_page(shop):
    o, site = _orch(shop, _llm(CART + [{"kind": "ask", "goal": "what is on the page"}],
                               verdict={"ok": True, "evidence": "Continue shopping", "reason": "on the page"},
                               answer="a Checkout button and Continue shopping"))
    rr = o.handle("open the cart and tell me what is on it", site)
    assert rr.status == "done" and rr.answer == "a Checkout button and Continue shopping"
    assert o.last_page["url"].endswith("/cart.html")                     # the ask did not navigate away


def test_an_unconfirmed_answer_is_replaced(shop):
    o, site = _orch(shop, _llm([{"kind": "ask", "goal": "cart"}],
                               verdict={"ok": True, "evidence": "free shipping on everything", "reason": "x"},
                               answer="free shipping"))
    rr = o.handle("does the cart page offer free shipping?", site)
    step = rr.steps[0]
    assert step["verified"]["ok"] is False and step["explanation"].startswith("Could not confirm")
    assert step["unconfirmed_answer"] == "free shipping" and rr.status == "failed"


def test_a_failed_automate_verify_is_replanned_once(shop, monkeypatch):
    o, site = _orch(shop, _llm(CART, verdict={"ok": False, "evidence": "", "reason": "this is not the cart"}))
    calls = []
    real = Navigator.execute_goal

    def spy(self, goal, *a, **kw):
        calls.append(goal)
        return real(self, goal, *a, **kw)

    monkeypatch.setattr(Navigator, "execute_goal", spy)
    rr = o.handle("take me to my cart", site)
    assert len(calls) == 2 and "this is not the cart" in calls[1]
    step = rr.steps[0]
    assert step["replanned_after_verify"] is True and step["verified"]["ok"] is False and rr.status == "failed"


def test_a_forced_kind_is_a_one_step_plan(shop):
    o, site = _orch(shop, _llm(CART))
    rr = o.handle("I like gifs", site, kind="remember")
    assert rr.plan.source == "override" and rr.steps[0]["kind"] == "remember" and rr.status == "done"
