"""Scripted demos: rehearsal actions -> an explanatory script -> one clean final take."""
import json
from dataclasses import replace

import pytest
from sitegraph.agents.curator import Curator
from sitegraph.agents.demo_agent import DemoAgent
from sitegraph.agents.navigator import Navigator
from sitegraph.blobs import BlobStore
from sitegraph.browser.session import BrowserSession
from sitegraph.db import device_key
from sitegraph.demo.script import DemoScript, Scene, ScriptError, build_script
from sitegraph.embed import HashEmbedder
from sitegraph.llm import ChatMessage, FakeLLM, ScriptedLLM, ToolCall
from sitegraph.normalize import domain_of
from sitegraph.policy import ExplorationPolicy
from sitegraph.schema import ensure_device_schema
from tests.helpers import site_responder

LOG = [
    {"id": "a1", "kind": "click", "label": "click 'About'", "element": {"gid": "el:about"}},     # a dead end
    {"id": "a2", "kind": "goto", "label": "open http://shop.test/index.html", "url": "http://shop.test/index.html"},
    {"id": "a3", "kind": "click", "label": "click 'Products'", "element": {"gid": "el:products"}},
    {"id": "a4", "kind": "fill", "label": "type into 'Search products'", "element": {"gid": "el:q"}, "value": "cap"},
]


def test_script_keeps_only_the_chosen_actions_in_order_and_reads_like_a_tutorial():
    script = build_script(
        title="Finding a product", goal="show how to find the caps", start_url="http://shop.test/index.html",
        intro="In this video you will see how to find a product in the Demo Shop.",
        outro="That's it: two steps to any product.",
        scenes=[{"action": "a3", "caption": "The Products page lists everything the shop sells, so start there."},
                {"action": "a4", "caption": "Type a word from the product name to narrow the list.",
                 "narration": "Search for cap.", "zoom": True},
                {"caption": "Here are the matching products."}],
        log=LOG)
    assert [s.action["id"] if s.action else None for s in script.scenes] == ["a3", "a4", None]   # a1 (dead end) gone
    md = script.to_markdown()
    assert md.startswith("# Finding a product")
    assert "## Intro" in md and "## Outro" in md and md.index("## Intro") < md.index("1.") < md.index("## Outro")
    assert "The Products page lists everything" in md and "click 'Products'" in md
    assert all(s.hold >= 2.5 for s in script.scenes)                 # every scene is held long enough to read
    again = DemoScript.from_json(script.to_json())
    assert again.to_markdown() == md and again.scenes[1].zoom is True


def test_script_rejects_unknown_actions_and_empty_scripts():
    with pytest.raises(ScriptError, match="a9"):
        build_script(title="t", goal="g", start_url="u", intro="", outro="",
                     scenes=[{"action": "a9", "caption": "x"}], log=LOG)
    with pytest.raises(ScriptError, match="at least one scene"):
        build_script(title="t", goal="g", start_url="u", intro="", outro="", scenes=[], log=LOG)


def test_typed_text_is_in_the_script_file_but_kept_out_of_the_storyboard_text():
    script = build_script(title="t", goal="g", start_url="u", intro="", outro="",
                          scenes=[{"action": "a4", "caption": "Search for it."}], log=LOG)
    assert json.loads(script.to_json())["scenes"][0]["action"]["value"] == "cap"   # needed to replay the typing
    assert "cap" not in script.scenes[0].caption and "cap" not in script.storyboard_caption(0)


# ---------- the whole flow in a real browser, with a scripted model ----------


@pytest.fixture(scope="module")
def shop(store, msettings, site_url, tmp_path_factory):
    domain = domain_of(site_url)
    key = device_key(msettings.uid, domain, "desktop")
    ensure_device_schema(store, key, 384)
    with BrowserSession() as s:
        Navigator(store, key, domain, s, msettings, HashEmbedder()).explore(
            site_url + "/index.html", ExplorationPolicy(max_depth=1))
    Curator(store, key, msettings, HashEmbedder(), FakeLLM(site_responder)).run_all(1)
    return store, msettings, key, domain, site_url, BlobStore(tmp_path_factory.mktemp("blobs"))


def call(name, args=None, id="c"):
    return ChatMessage("assistant", tool_calls=[ToolCall(id, name, args or {})])


def say(text):
    return ChatMessage("assistant", content=text)


def click_named(name, id):
    """Scripted step: click the element called `name` from the last observe result."""
    def step(messages, tools):
        obs = next(m for m in reversed(messages) if m.role == "tool" and m.name == "observe")
        data = json.loads("\n".join(l for l in obs.content.splitlines() if l.startswith("{")))["data"]
        return call("click", {"element_id": next(e["id"] for e in data["elements"] if e["name"] == name)}, id)
    return step


SCRIPT = {"title": "Browsing the Demo Shop", "intro": "You will see where to find every product of the shop.",
          "outro": "Now you know where the whole catalogue lives.",
          "scenes": [{"action": "a3", "caption": "The full catalogue sits behind Browse all products, on the home page."},
                     {"caption": "Every product is listed here, each with its own page."}]}


def _agent(shop, llm, **settings):
    store, base, key, domain, _, blobs = shop
    return DemoAgent(replace(base, **settings), store, HashEmbedder(), FakeLLM(site_responder), blobs, key, domain,
                     chat_llm=llm)


@pytest.mark.integration
@pytest.mark.browser
@pytest.mark.slow
def test_rehearse_script_approve_and_record_one_clean_take(shop):
    store, _, key, _, site_url, blobs = shop
    llm = ScriptedLLM([
        call("observe", id="c1"), click_named("Contact", "c2"),                       # a dead end in the rehearsal
        call("open_url", {"url": site_url + "/index.html"}, "c3"),                   # back to the start
        call("observe", id="c4"), click_named("Browse all products", "c5"),          # the way that works
        call("rehearsal_log", id="c6"),
        call("write_script", SCRIPT, "c7"),                                          # pauses for approval
        call("record_final", id="c8"),
        call("review_take", {"scene": 1}, "c9"),
        call("render", {"formats": ["video", "guide"]}, "c10"),
        say("Made a video and a guide of browsing the shop.")])
    agent = _agent(shop, llm, demo_approve_script=True)
    try:
        first = agent.run("make a video showing where to find all products")
        assert first.status == "needs_input" and "approve" in first.answer.lower()
        assert "Browse all products" in first.answer and "## Intro" in first.answer      # the script is shown
        log = agent.rehearsal_log()
        assert [e["id"] for e in log][:3] == ["a1", "a2", "a3"] and "Contact" in log[0]["action"]
        done = agent.reply("approve")
    finally:
        agent.close()
    assert done.status == "done", done.answer
    formats = {f for f, _ in done.renders}
    assert {"video", "guide", "guide_md"} <= formats
    assert len(agent.take.segments) == 1                                              # one continuous take
    assert set(agent.take.keyframes) == {0, 1, 2, 3}                                  # title, two scenes, closing
    assert [s.action["id"] if s.action else None for s in agent.script.scenes] == ["a3", None]   # no dead end
    md_ref = store.ro_query(key, "MATCH (sb:Storyboard {gid:$g}) RETURN sb.script_ref", {"g": done.storyboard_gid})[0][0]
    assert blobs.read(md_ref).decode().startswith("# Browsing the Demo Shop")
    assert llm.images[-2] is not None                                                 # review_take showed a frame


@pytest.mark.integration
@pytest.mark.browser
def test_requested_changes_block_recording_until_the_script_is_rewritten(shop):
    llm = ScriptedLLM([
        call("observe", id="c1"), click_named("Browse all products", "c2"),
        call("write_script", {**SCRIPT, "scenes": [{"action": "a1", "caption": "Open the catalogue."}]}, "c3"),
        call("record_final", id="c4"),
        say("The user wants changes; I will rewrite the script.")])
    agent = _agent(shop, llm, demo_approve_script=True)
    try:
        assert agent.run("make a video of the catalogue").status == "needs_input"
        agent.reply("make the captions explain why, please")
    finally:
        agent.close()
    refused = next(m for m in llm.calls[-1][0] if m.role == "tool" and m.name == "record_final")
    assert "asked for changes" in refused.content and agent.take is None
