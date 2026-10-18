import json

import pytest

from sitegraph.agents.bootstrap import SiteProfile
from sitegraph.agents.planner import Plan, Planner, PlanStep, choose_target, input_vars, needs_choice
from sitegraph.graph.queries import Target
from sitegraph.llm import FakeLLM, OfflineLLM, TracingLLM


def _steps_llm(steps):
    return FakeLLM(lambda s, u: json.dumps({"steps": steps}))


def test_plan_splits_a_compound_request():
    llm = _steps_llm([{"kind": "automate", "goal": "Academic category"},
                      {"kind": "automate", "goal": "book Logan Kade", "item": "Logan Kade"},
                      {"kind": "ask", "goal": "price"}])
    site = SiteProfile(purpose="an online book shop", sections=[{"name": "Academic", "url": "/academic"}])
    plan = Planner(llm, None, site).plan("price of Logan Kade in Academic")
    assert plan.source == "model" and [s.kind for s in plan.steps] == ["automate", "automate", "ask"]
    assert plan.steps[1] == PlanStep("automate", "book Logan Kade", {}, "Logan Kade")
    system, user = llm.calls[0]
    assert "price of Logan Kade in Academic" in user and "online book shop" in user and "Academic" in user
    assert "data" in system.lower()                    # page / site text is data, not instructions


def test_plan_without_a_model_is_one_keyword_step(tmp_path):
    for llm in (None, OfflineLLM(), TracingLLM(OfflineLLM(), tmp_path)):
        plan = Planner(llm, None, None).plan("explore the site")
        assert plan == Plan([PlanStep("explore", "explore the site", {}, None)], "keywords")
    assert Planner(None, None, None).plan("What is the price of the red mug?").steps[0].kind == "ask"


def test_plan_falls_back_to_keywords_on_junk():
    for reply in ("no idea", '{"steps": "lots"}', '{"steps": []}', '[1, 2]', '{"steps": [{"kind": "dance"}]}'):
        plan = Planner(FakeLLM(lambda s, u, r=reply: r), None, None).plan("open the cart")
        assert plan.source == "keywords" and [s.kind for s in plan.steps] == ["automate"], reply


def test_plan_is_capped_and_unknown_kinds_dropped():
    steps = [{"kind": "automate", "goal": f"page {i}"} for i in range(8)]
    steps.insert(2, {"kind": "dance", "goal": "boogie"})
    plan = Planner(_steps_llm(steps), None, None).plan("visit many pages")
    assert len(plan.steps) == 6 and all(s.kind == "automate" for s in plan.steps)
    assert [s.goal for s in plan.steps] == [f"page {i}" for i in range(6)]
    assert len(Planner(_steps_llm(steps), None, None, max_steps=3).plan("x").steps) == 3


def test_demo_and_remember_requests_stay_one_step():
    demo = _steps_llm([{"kind": "automate", "goal": "checkout"}, {"kind": "demo", "goal": "checkout demo"}])
    plan = Planner(demo, None, None).plan("make a gif of the checkout")
    assert [s.kind for s in plan.steps] == ["demo"]
    mem = _steps_llm([{"kind": "remember", "goal": "I like jazz"}, {"kind": "recommend", "goal": "x"}])
    assert [s.kind for s in Planner(mem, None, None).plan("remember that I like jazz").steps] == ["remember"]


def test_plan_maps_params_onto_known_input_variables():
    llm = _steps_llm([{"kind": "automate", "goal": "search results", "params": {"Q": "mystery", "Category": "Mystery",
                                                                                  "colour": "red"}}])
    plan = Planner(llm, None, None).plan("mystery books in stock", inputs=["q", "category"])
    assert plan.steps[0].params == {"q": "mystery", "category": "Mystery", "colour": "red"}
    assert "q, category" in llm.calls[0][1]


def test_needs_choice_only_for_close_targets():
    a, b = Target("State", "s1", "Cart", 0.80), Target("State", "s2", "Basket", 0.78)
    assert needs_choice([a, b], 0.05)
    assert not needs_choice([a, Target("State", "s2", "Basket", 0.60)], 0.05)    # a clear winner needs no model
    assert not needs_choice([a], 0.05) and not needs_choice([], 0.05)


def test_choose_target_without_a_model_or_with_junk_returns_none():
    cands = [Target("State", "s1", "Cart", 0.80), Target("State", "s2", "Basket", 0.78)]
    assert choose_target(OfflineLLM(), "open the cart", cands) is None
    assert choose_target(None, "open the cart", cands) is None
    for reply in ("nope", '{"pick": 7}', '{"pick": "x"}', '{"pick": 0}'):
        assert choose_target(FakeLLM(lambda s, u, r=reply: r), "open the cart", cands) is None, reply


def _seed(store, gkey):
    store.query(gkey, """CREATE (:Intent {gid:'in1', text:'open cart'})-[:FULFILLED_BY {score:1.0}]->(:State {gid:'s1'}),
                                (:Intent {gid:'in2', text:'open basket'})-[:FULFILLED_BY {score:1.0}]->(:State {gid:'s2'}),
                                (:State {gid:'s3'})""")


def _score(store, gkey, ig, sg):
    return store.ro_query(gkey, "MATCH (:Intent {gid:$i})-[f:FULFILLED_BY]->({gid:$s}) RETURN f.score",
                          {"i": ig, "s": sg})[0][0]


@pytest.mark.integration
def test_the_model_choice_is_cached_on_the_intents(store, gkey):
    _seed(store, gkey)
    cands = [Target("State", "s1", "Cart page", 0.80, intent="in1"),
             Target("State", "s2", "Basket page", 0.78, intent="in2"),
             Target("State", "s3", "About page", 0.77)]
    llm = FakeLLM(lambda s, u: '{"pick": 2, "reason": "basket holds the items"}')
    chosen = choose_target(llm, "show my basket", cands, store=store, key=gkey)
    assert chosen is cands[1] and len(llm.calls) == 1
    assert "Basket page" in llm.calls[0][1] and "show my basket" in llm.calls[0][1]
    assert _score(store, gkey, "in2", "s2") == pytest.approx(1.0)          # capped at 1.0
    assert _score(store, gkey, "in1", "s1") == pytest.approx(0.9)          # the others lose 0.1
    choose_target(llm, "show my basket", cands, store=store, key=gkey)
    assert _score(store, gkey, "in1", "s1") == pytest.approx(0.8)


@pytest.mark.integration
def test_input_vars_lists_route_variables(store, gkey):
    store.query(gkey, """CREATE (a:State {gid:'a'}), (b:State {gid:'b'}), (c:State {gid:'c'}),
                                (a)-[:ACTION {gid:'x1', requires_inputs:['q']}]->(b),
                                (b)-[:ACTION {gid:'x2', value_var:'category', requires_inputs:['q']}]->(c),
                                (a)-[:ACTION {gid:'x3'}]->(c)""")
    assert input_vars(store, gkey) == ["category", "q"]


def _plan_of(steps, text):
    return Planner(FakeLLM(lambda s, u: json.dumps({"steps": steps})), max_steps=6).plan(text)


def test_an_explore_step_the_request_never_asked_for_is_dropped():
    # a small model labels a plain question "explore": exploring is expensive, so it needs the user's word for it
    plan = _plan_of([{"kind": "explore", "goal": "cheapest book"}], "so what is the cheapest book?")
    assert [s.kind for s in plan.steps] == ["ask"]                       # the keyword fallback for a question
    plan = _plan_of([{"kind": "explore", "goal": "site"}, {"kind": "ask", "goal": "cheapest book"}],
                    "what is the cheapest book?")
    assert [s.kind for s in plan.steps] == ["ask"]


def test_repeated_explore_steps_become_one():
    plan = _plan_of([{"kind": "explore", "goal": "site"}, {"kind": "explore", "goal": "philosophy"},
                     {"kind": "ask", "goal": "cheapest"}], "explore the site and find the cheapest philosophy book")
    assert [s.kind for s in plan.steps] == ["explore", "ask"]
