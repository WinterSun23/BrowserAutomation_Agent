import json

import pytest
from sitegraph.agents.curator import Curator
from sitegraph.agents.demo_agent import DemoAgent
from sitegraph.agents.navigator import Navigator
from sitegraph.blobs import BlobStore
from sitegraph.browser.session import BrowserSession
from sitegraph.db import device_key
from sitegraph.demo.director import DemoResult
from sitegraph.demo.spec import DemoSpec
from sitegraph.embed import HashEmbedder
from sitegraph.llm import ChatMessage, FakeLLM, OfflineLLM, ScriptedLLM, ToolCall
from sitegraph.normalize import domain_of
from sitegraph.policy import ExplorationPolicy
from sitegraph.schema import ensure_device_schema
from tests.helpers import site_responder

pytestmark = [pytest.mark.integration, pytest.mark.browser, pytest.mark.slow]


@pytest.fixture(scope="module")
def shop(store, msettings, site_url, tmp_path_factory):
    domain = domain_of(site_url)
    key = device_key(msettings.uid, domain, "desktop")
    ensure_device_schema(store, key, 384)
    with BrowserSession() as s:
        Navigator(store, key, domain, s, msettings, HashEmbedder()).explore(
            site_url + "/index.html", ExplorationPolicy(max_depth=2))
    Curator(store, key, msettings, HashEmbedder(), FakeLLM(site_responder)).run_all(1)
    return store, msettings, key, domain, site_url, BlobStore(tmp_path_factory.mktemp("blobs"))


def call(name, args=None, id="c"):
    return ChatMessage("assistant", tool_calls=[ToolCall(id, name, args or {})])


def say(text):
    return ChatMessage("assistant", content=text)


def body(message: ChatMessage) -> dict:
    """The JSON of a tool message (untrusted-wrapped or not)."""
    return json.loads("\n".join(l for l in message.content.splitlines() if l.startswith("{")))


def last_tool(messages, name):
    return next(m for m in reversed(messages) if m.role == "tool" and m.name == name)


def _agent(shop, llm):
    store, settings, key, domain, _, blobs = shop
    return DemoAgent(settings, store, HashEmbedder(), FakeLLM(site_responder), blobs, key, domain, chat_llm=llm)


def test_agent_builds_a_gif_and_a_screenshot_through_tools(shop):
    store, _, key, _, _, blobs = shop
    llm = ScriptedLLM([
        call("find_screens", {"query": "open cart"}, "c1"),
        lambda m, t: call("plan_route", {"to_state": body(m[-1])["data"][0]["gid"]}, "c2"),
        lambda m, t: call("capture_route", {"route_id": body(m[-1])["data"]["route_id"], "visible": True}, "c3"),
        lambda m, t: call("preview_frame", {"shot_id": body(m[-1])["data"]["shots"][0]["shot_id"]}, "c4"),
        lambda m, t: call("compose_storyboard", {"shot_ids": [s["shot_id"] for s in body(last_tool(m, "capture_route"))
                                                              ["data"]["shots"]], "title": "Opening the cart"}, "c5"),
        call("render", {"formats": ["gif", "screenshot"]}, "c6"),
        say("Made a gif and a screenshot of opening the cart.")])
    agent = _agent(shop, llm)
    try:
        res = agent.run("make a gif of opening the cart")
    finally:
        agent.close()
    assert res.status == "done" and {f for f, _ in res.renders} == {"gif", "screenshot"}
    assert "gif" in res.answer
    for _, ref in res.renders:
        assert blobs.path(ref).stat().st_size > 0
    [(label, png)] = llm.images[4]                                                # the preview reached the model
    assert label.startswith("shot 0") and png[:4] == b"\x89PNG"
    assert store.ro_query(key, "MATCH (:Storyboard {gid:$g})-[:RENDERED_AS]->(r:Render) RETURN r.status ORDER BY r.format",
                          {"g": res.storyboard_gid}) == [["ok"], ["ok"]]
    assert store.ro_query(key, "MATCH (:Storyboard {gid:$g})<-[:IN_STORYBOARD]-(:Shot)-[:FROM_STEP]->(:Step) "
                               "RETURN count(*) > 0", {"g": res.storyboard_gid}) == [[True]]


def test_agent_retakes_a_shot_it_judges_bad(shop):
    store, _, key, _, _, _ = shop
    llm = ScriptedLLM([
        call("capture_still", {"caption": "Home pgae"}, "c1"),
        call("preview_frame", {"shot_id": 0}, "c2"),
        call("discard_shot", {"shot_id": 0}, "c3"),
        call("capture_still", {"caption": "Home page"}, "c4"),
        call("compose_storyboard", {"shot_ids": [1], "title": "Home"}, "c5"),
        call("render", {"formats": ["screenshot"]}, "c6"),
        say("Retook the first shot (typo in the caption) and rendered a screenshot.")])
    agent = _agent(shop, llm)
    try:
        res = agent.run("a screenshot of the home page")
    finally:
        agent.close()
    assert res.status == "done" and [f for f, _ in res.renders] == ["screenshot"]
    assert store.ro_query(key, "MATCH (:Storyboard {gid:$g})<-[:IN_STORYBOARD]-(sh:Shot) RETURN sh.caption",
                          {"g": res.storyboard_gid}) == [["Home page"]]          # only the retake made it into the demo


def test_agent_can_ask_the_user_and_continue_with_the_answer(shop):
    llm = ScriptedLLM([
        call("ask_user", {"question": "Which format would you like: gif or screenshot?"}, "c1"),
        call("capture_still", {"caption": "Home"}, "c2"),
        call("compose_storyboard", {"shot_ids": [0], "title": "Home"}, "c3"),
        call("render", {"formats": ["screenshot"]}, "c4"),
        say("Rendered a screenshot, as you chose.")])
    agent = _agent(shop, llm)
    try:
        first = agent.run("make a demo of the home page")
        assert first.status == "needs_input" and "gif or screenshot" in first.answer
        assert agent.session.is_open                                              # still recording, waiting for the answer
        second = agent.reply("screenshot please")
    finally:
        agent.close()
    assert second.status == "done" and [f for f, _ in second.renders] == ["screenshot"]


def test_a_demo_never_takes_a_risky_action_without_approval(shop):
    def pick_checkout(messages, tools):
        elements = body(messages[-1])["data"]["elements"]
        return call("capture_interaction", {"element_id": next(e["id"] for e in elements if e["name"] == "Checkout"),
                                            "caption": "Check out"}, "c2")

    llm = ScriptedLLM([call("observe", id="c1"), pick_checkout])
    agent = _agent(shop, llm)
    try:
        res = agent.run("demo the checkout", start_url=shop[4] + "/cart.html")
    finally:
        agent.close()
    assert res.status == "blocked" and "approval" in res.answer and res.renders == []


def test_render_before_compose_is_an_error_the_agent_sees(shop):
    llm = ScriptedLLM([call("render", {"formats": ["gif"]}, "c1"), say("I could not render yet.")])
    agent = _agent(shop, llm)
    try:
        agent.run("a gif")
        seen = last_tool(agent.history, "render")
    finally:
        agent.close()
    assert "compose_storyboard first" in seen.content


def test_without_a_tool_calling_model_the_fixed_pipeline_is_used(shop, monkeypatch):
    from sitegraph.agents import demo_agent

    used = {}

    class StubDirector:
        def __init__(self, *a, **k): ...
        def run(self, text, start_url=None):
            used["text"] = text
            return DemoResult("sb:1", [("video", "x.mp4")], DemoSpec(["video"], "goal", text))

    monkeypatch.setattr(demo_agent, "DemoDirector", StubDirector)
    store, settings, key, domain, _, blobs = shop
    agent = DemoAgent(settings, store, HashEmbedder(), FakeLLM(site_responder), blobs, key, domain, chat_llm=OfflineLLM())
    res = agent.run("a video of the shop")
    assert used["text"] == "a video of the shop" and res.renders == [("video", "x.mp4")] and agent.session is None
