import json

from sitegraph.agents.finish import Finish, finish_check, wants_answer
from sitegraph.agents.orchestrator import RequestResult
from sitegraph.agents.planner import Plan, PlanStep
from sitegraph.llm import FakeLLM, OfflineLLM


def _rr(text, steps, plan_steps=None):
    plan = Plan(plan_steps or [PlanStep(s["kind"], s.get("goal", "x")) for s in steps], "model")
    return RequestResult(text, plan, [dict(s, step=i) for i, s in enumerate(steps)], "done")


PAGE = {"url": "http://shop.test/category/philosophy", "text": "Sapiens 54.23 Pound. The Gift 23.61 Pound. Plato 45.00"}
ASK_STEPS = [{"kind": "automate", "goal": "philosophy", "status": "done", "explanation": "click Philosophy"},
             {"kind": "ask", "goal": "cheapest", "status": "done", "explanation": "The Gift at 23.61"}]


def _llm(reply, seen=None):
    def r(system, user):
        if seen is not None:
            seen.append((system, user))
        return reply if isinstance(reply, str) else json.dumps(reply)
    return FakeLLM(r)


def test_the_word_list_is_only_the_fallback():
    assert wants_answer("what is the cheapest book?") and wants_answer("tell me the price")
    assert not wants_answer("take me to my cart")


def test_without_a_model_the_word_list_decides():
    rr = _rr("take me to my cart", [{"kind": "automate", "status": "done"}])
    f = finish_check("take me to my cart", rr, PAGE, OfflineLLM())
    assert f.complete is True and f.how == "words" and f.answer is None
    rr = _rr("find the cheapest book", [{"kind": "explore", "status": "done"}])
    f = finish_check("find the cheapest book", rr, PAGE, OfflineLLM())
    assert f.complete is False and f.how == "words"


def test_a_complete_verdict_with_a_supported_answer_is_trusted():
    llm = _llm({"complete": True, "answer": "The Gift, 23.61 Pound", "evidence": "The Gift 23.61 Pound",
                "reason": "the category page lists prices"})
    f = finish_check("find the cheapest philosophy book", _rr("q", ASK_STEPS), PAGE, llm)
    assert f == Finish(True, "The Gift, 23.61 Pound", "The Gift 23.61 Pound",
                       "the category page lists prices", "model")


def test_an_answer_whose_quote_is_not_in_the_results_is_not_complete():
    llm = _llm({"complete": True, "answer": "Plato, 5.00", "evidence": "Plato 5.00 Pound", "reason": "x"})
    f = finish_check("find the cheapest philosophy book", _rr("q", ASK_STEPS), PAGE, llm)
    assert f.complete is False and "not supported" in f.reason and f.how == "model"


def test_a_pure_action_can_be_complete_without_an_answer():
    llm = _llm({"complete": True, "answer": None, "evidence": "", "reason": "the cart is open"})
    rr = _rr("take me to my cart", [{"kind": "automate", "status": "done", "explanation": "click Cart"}])
    f = finish_check("take me to my cart", rr, PAGE, llm)
    assert f.complete is True and f.answer is None and f.how == "model"


def test_the_model_can_say_more_is_needed():
    llm = _llm({"complete": False, "answer": None, "evidence": "", "reason": "only the home page was read"})
    rr = _rr("find the cheapest philosophy book", [{"kind": "explore", "status": "done"}])
    f = finish_check("find the cheapest philosophy book", rr, PAGE, llm)
    assert f.complete is False and f.reason == "only the home page was read"


def test_junk_output_falls_back_to_the_word_list():
    rr = _rr("find the cheapest book", [{"kind": "explore", "status": "done"}])
    f = finish_check("find the cheapest book", rr, PAGE, _llm("I think it is fine!"))
    assert f.complete is False and f.how == "words"
    rr = _rr("take me to my cart", [{"kind": "automate", "status": "done"}])
    assert finish_check("take me to my cart", rr, PAGE, _llm("{}")).complete is True


def test_the_model_sees_the_steps_and_the_page_as_data():
    seen = []
    llm = _llm({"complete": False, "answer": None, "evidence": "", "reason": "r"}, seen)
    finish_check("find the cheapest philosophy book", _rr("q", ASK_STEPS), PAGE, llm)
    system, user = seen[0]
    assert "data" in system.lower() and "decide whether" in system.lower()
    assert "find the cheapest philosophy book" in user and "click Philosophy" in user
    assert "http://shop.test/category/philosophy" in user and "The Gift 23.61 Pound" in user
