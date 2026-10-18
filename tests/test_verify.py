import json

from sitegraph.agents.verify import Verdict, verify
from sitegraph.llm import FakeLLM, OfflineLLM

PAGE = "Books to Scrape\nA Light in the Attic\n£51.77\n  In   stock (22 available)\nAdd to basket"


def _llm(reply):
    return FakeLLM(lambda s, u: reply if isinstance(reply, str) else json.dumps(reply))


def test_reached_state_without_a_model_is_ok_by_state():
    for llm in (None, OfflineLLM()):
        v = verify("open the book A Light in the Attic", "automate", True, PAGE, llm)
        assert v == Verdict(True, "", "state", v.reason) and v.reason
        bad = verify("open the book", "automate", False, PAGE, llm)
        assert not bad.ok and bad.how == "none"
    by_url = verify("open the book", "automate", False, PAGE, None, url_ok=True)
    assert by_url.ok and by_url.how == "url"


def test_model_evidence_must_be_on_the_page():
    v = verify("price of A Light in the Attic", "ask", True, PAGE,
               _llm({"ok": True, "evidence": "£12.00 special offer"}), answer="£12.00")
    assert not v.ok and v.how == "model" and v.reason == "evidence not found on the page"
    good = verify("is A Light in the Attic in stock?", "ask", True, PAGE,
                  _llm({"ok": True, "evidence": "in stock (22   AVAILABLE)"}), answer="yes, 22")
    assert good.ok and good.how == "model" and good.evidence == "in stock (22   AVAILABLE)"   # whitespace and case


def test_the_model_can_reject_and_sees_request_answer_and_page_as_data():
    llm = _llm({"ok": False, "evidence": "", "reason": "this is the basket, not the book"})
    v = verify("open the book", "automate", True, PAGE, llm)
    assert not v.ok and v.how == "model" and "basket" in v.reason
    system, user = llm.calls[0]
    assert "open the book" in user and "A Light in the Attic" in user and "data" in system.lower()
    verify("price?", "ask", True, PAGE, llm, answer="£51.77")
    assert "£51.77" in llm.calls[1][1]


def test_a_failed_deterministic_check_needs_no_model():
    llm = _llm({"ok": True, "evidence": "Add to basket"})
    v = verify("open the book", "automate", False, PAGE, llm)
    assert not v.ok and v.how == "none" and llm.calls == []


def test_junk_verdict_counts_as_unverified_not_crash():
    for reply in ("I think so", "[]", '{"ok": "maybe"}', '{"evidence": "Add to basket"}', "", '{"ok": true}'):
        v = verify("open the book", "automate", True, PAGE, _llm(reply))
        assert isinstance(v, Verdict) and not v.ok, reply
    boom = FakeLLM(lambda s, u: (_ for _ in ()).throw(RuntimeError("model down")))
    v = verify("open the book", "automate", True, PAGE, boom)
    assert not v.ok and v.reason


def test_verdict_as_dict():
    assert Verdict(True, "x", "model", "r").to_dict() == {"ok": True, "evidence": "x", "how": "model", "reason": "r"}
