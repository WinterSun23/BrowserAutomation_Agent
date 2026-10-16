import json
import pytest
from sitegraph.agents.curator import Curator
from sitegraph.agents.navigator import Navigator
from sitegraph.browser.session import BrowserSession
from sitegraph.db import device_key
from sitegraph.demo.spec import interpret
from sitegraph.demo.storyboard import StoryboardBuilder
from sitegraph.embed import HashEmbedder
from sitegraph.llm import FakeLLM
from sitegraph.normalize import domain_of
from sitegraph.policy import ExplorationPolicy
from sitegraph.schema import ensure_device_schema
from tests.helpers import site_responder


def test_interpret_parses_and_validates():
    llm = FakeLLM(lambda s, u: json.dumps({"formats": ["gif", "hologram"], "scope_kind": "transition",
                                           "target": "account menu", "max_seconds": 10}))
    spec = interpret(llm, "gif of the account menu opening")
    assert spec.formats == ["gif"] and spec.scope_kind == "transition" and spec.target == "account menu"


def test_interpret_defaults_on_garbage():
    spec = interpret(FakeLLM(lambda s, u: "lol no"), "show me checkout", {"demo.format": "guide"})
    assert spec.formats == ["guide"] and spec.scope_kind == "goal" and spec.target == "show me checkout"


def test_interpret_without_llm_honours_formats_named_in_the_request():
    offline = FakeLLM(lambda s, u: "lol no")
    assert interpret(offline, "make a gif of the menu").formats == ["gif"]
    assert interpret(offline, "a screenshot and a step by step guide of checkout").formats == ["screenshot", "guide"]
    assert interpret(offline, "show me the cart", {"demo.format": "guide"}).formats == ["guide"]   # preference still wins


@pytest.fixture(scope="module")
def graph(store, msettings, site_url):
    settings = msettings
    domain = domain_of(site_url)
    key = device_key(settings.uid, domain, "desktop")
    ensure_device_schema(store, key, 384)
    with BrowserSession() as s:
        Navigator(store, key, domain, s, settings, HashEmbedder()).explore(site_url + "/index.html",
                                                                           ExplorationPolicy(max_depth=2))
    Curator(store, key, settings, HashEmbedder(), FakeLLM(site_responder)).run_all(1)
    return store, key


@pytest.mark.integration
@pytest.mark.browser
def test_goal_storyboard_persists_shots(graph):
    store, key = graph
    b = StoryboardBuilder(store, key, HashEmbedder(), FakeLLM(site_responder))
    sb = b.build(interpret(FakeLLM(lambda s, u: json.dumps({"formats": ["video"], "scope_kind": "goal",
                                                            "target": "open cart"})), "video of opening the cart"),
                 "video of opening the cart")
    assert sb.shots and all(not s.hidden for s in sb.shots) and sb.shots[0].caption == "Step 1"
    assert store.ro_query(key, "MATCH (:DemoRequest)-[:PRODUCED]->(sb:Storyboard {gid:$g})<-[:IN_STORYBOARD]-(sh:Shot) "
                               "RETURN count(sh)", {"g": sb.gid}) == [[len(sb.shots)]]


@pytest.mark.integration
@pytest.mark.browser
def test_transition_storyboard(graph):
    store, key = graph
    b = StoryboardBuilder(store, key, HashEmbedder(), FakeLLM(site_responder))
    spec = interpret(FakeLLM(lambda s, u: json.dumps({"formats": ["gif"], "scope_kind": "transition",
                                                      "target": "account menu"})), "gif of the account menu")
    sb = b.build(spec, "gif of the account menu")
    visible = [s for s in sb.shots if not s.hidden]
    assert visible[0].kind == "micro" and visible[0].transition is not None
    assert visible[-1].kind == "still"
