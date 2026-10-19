import json

import pytest
from sitegraph.agents.curator import Curator
from sitegraph.agents.execution_agent import ExecutionAgent
from sitegraph.agents.navigator import Navigator
from sitegraph.agents.orchestrator import Orchestrator, RequestResult
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


def _llm(steps, verdict=None, answer="nothing", finish=None):
    def r(system, user):
        if "decide whether" in system.lower():
            return json.dumps(finish) if finish is not None else "not json"      # None: the word list decides
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


def test_a_question_the_plan_did_not_answer_goes_on_to_the_tool_loop(shop):
    # the plan only explored; "find the cheapest ..." still needs an answer, so the model finishes it
    o, site = _orch(shop, _llm([{"kind": "explore", "goal": "site"}]))
    seen = []
    loop_llm = ScriptedLLM([lambda m, t: seen.append(m[-1].content) or ChatMessage("assistant", content="the cheapest is X")])
    agent = ExecutionAgent(o.s, o.store, o.embedder, o.llm, None, site, o.profile, chat_llm=loop_llm, orchestrator=o)
    try:
        run = agent.ask("explore and find the cheapest product")
        assert run.answer == "the cheapest is X" and len(loop_llm.calls) == 1
        assert "Explored" in seen[0] and "finish the request" in seen[0].lower()
    finally:
        agent.close()


def test_a_finished_action_with_no_question_needs_no_tool_loop(shop):
    o, site = _orch(shop, _llm(CART))
    loop_llm = ScriptedLLM([])
    agent = ExecutionAgent(o.s, o.store, o.embedder, o.llm, None, site, o.profile, chat_llm=loop_llm, orchestrator=o)
    try:
        assert agent.ask("take me to my cart").status == "done" and not loop_llm.calls
    finally:
        agent.close()


class _Counting:
    """Wraps an LLM and counts calls the way TracingLLM.counters() reports them."""

    def __init__(self, inner):
        self.inner, self.calls = inner, 0

    def complete(self, system, user, image_png=None):
        self.calls += 1
        return self.inner.complete(system, user, image_png)

    def counters(self):
        return {"calls": self.calls, "tokens": self.calls * 100}


def test_the_plans_own_model_calls_are_counted_and_the_summary_says_where_it_ended(shop):
    o, site = _orch(shop, _Counting(_llm(CART)))
    agent = ExecutionAgent(o.s, o.store, o.embedder, o.llm, None, site, o.profile, chat_llm=ScriptedLLM([]),
                           orchestrator=o)
    try:
        run = agent.ask("take me to my cart")
        assert run.status == "done" and run.steps >= 2 and run.tokens == run.steps * 100   # planner + verify calls
        assert "cart.html" in run.answer and "automate" not in run.answer.lower().split(":")[0]
    finally:
        agent.close()


def test_the_answer_comes_from_ask_steps_only(shop):
    o, site = _orch(shop, _llm(CART))
    rr = o.handle("take me to my cart", site)
    assert rr.status == "done" and rr.answer is None            # the route text of an automate step is not an answer


def test_chat_does_not_force_goal_exploration(shop, monkeypatch):
    o, site = _orch(shop, _llm([{"kind": "automate", "goal": "zzqx vvkk"}]))
    o.s.goal_explore = True
    calls = []
    monkeypatch.setattr(Orchestrator, "explore_toward", lambda self, *a, **k: calls.append(a) or None)
    try:
        rr = o.handle("zzqx vvkk", site, goal_explore=False)
        assert not calls and rr.steps[0]["status"] == "no_target" and "goal_explore" not in rr.steps[0]
        o.handle("zzqx vvkk", site)                                # run and eval keep the automatic search
        assert len(calls) == 1
    finally:
        o.s.goal_explore = False


def test_finish_check_returns_the_answer_without_starting_the_loop(shop):
    fin = {"complete": True, "answer": "Your cart is empty", "evidence": "Your cart", "reason": "cart page read"}
    o, site = _orch(shop, _Counting(_llm(CART, finish=fin)))
    loop_llm = ScriptedLLM([])
    agent = ExecutionAgent(o.s, o.store, o.embedder, o.llm, None, site, o.profile, chat_llm=loop_llm, orchestrator=o)
    try:
        run = agent.ask("what is in my cart?")
        assert run.status == "done" and run.answer == "Your cart is empty" and not loop_llm.calls
        assert run.steps >= 3                                      # planner, verify and the finish check are counted
    finally:
        agent.close()


def test_finish_check_hands_unfinished_requests_to_the_loop_with_its_reason(shop):
    fin = {"complete": False, "answer": None, "evidence": "", "reason": "only the cart was opened, prices not read"}
    o, site = _orch(shop, _llm(CART, finish=fin))
    seen = []
    loop_llm = ScriptedLLM([lambda m, t: seen.append(m[-1].content) or ChatMessage("assistant", content="done it")])
    agent = ExecutionAgent(o.s, o.store, o.embedder, o.llm, None, site, o.profile, chat_llm=loop_llm, orchestrator=o)
    try:
        assert agent.ask("open the cart and tell me the total").answer == "done it"
        assert "prices not read" in seen[0] and "finish the request" in seen[0].lower()
    finally:
        agent.close()


def test_other_kinds_print_their_real_result(shop):
    from sitegraph.agents.planner import Plan, PlanStep

    o, site = _orch(shop, _llm(CART))
    agent = ExecutionAgent(o.s, o.store, o.embedder, o.llm, None, site, o.profile, chat_llm=ScriptedLLM([]),
                           orchestrator=o)
    try:
        def summary(text, kind, result):
            rr = RequestResult(text, Plan([PlanStep(kind, "g")], "model"), [dict(result, kind=kind, step=0)], "done")
            return agent._summary(rr)

        assert "I like gifs" in summary("remember that I like gifs", "remember", {"memory": "mem:1"})
        rec = summary("what should I automate?", "recommend", {
            "next_actions": [{"label": "Open the cart"}], "suggested_workflows": [], "demo_candidates": [],
            "frontier": [], "workflows": [{"name": "Checkout flow"}]})
        assert "Open the cart" in rec and "Checkout flow" in rec
        assert "workflow" in summary("save this as a workflow", "save_workflow", {"workflow": "wf:1"}).lower()
    finally:
        agent.close()
