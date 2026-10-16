import json
import pytest
from sitegraph.agents.orchestrator import Orchestrator
from sitegraph.blobs import BlobStore
from sitegraph.embed import HashEmbedder
from sitegraph.events import NullEvents
from sitegraph.llm import FakeLLM
from sitegraph.ops import Ops
from tests.helpers import site_responder


def _llm(extra=None):
    def r(system, user):
        if "Classify" in user:
            t = user.split("\n", 1)[0].lower()   # only the request line: the rest of the prompt lists every kind
            if "explore" in t:
                return json.dumps({"kind": "explore", **(extra or {})})
            return json.dumps({"kind": "automate", "goal": "open cart", "params": {}})
        return site_responder(system, user)
    return FakeLLM(r)


def test_classify_falls_back_to_keywords(store, settings):
    o = Orchestrator(settings, store, HashEmbedder(), FakeLLM(lambda s, u: "nope"), BlobStore(settings.blob_dir),
                     NullEvents())
    assert o.classify("make a gif of the menu")["kind"] == "demo"
    assert o.classify("explore the site")["kind"] == "explore"
    assert o.classify("what should I automate?")["kind"] == "recommend"
    assert o.classify("go to my cart")["kind"] == "automate"


@pytest.mark.integration
@pytest.mark.browser
def test_explore_then_automate_end_to_end(store, settings, site_url):
    settings.min_target_score = 0.3
    o = Orchestrator(settings, store, HashEmbedder(), _llm(), BlobStore(settings.blob_dir), NullEvents())
    site = o.register_site(site_url + "/index.html")
    ops = Ops(store, settings.uid, site.domain)
    t1 = ops.create_task("chat", "explore the site")
    r1 = o.run_task(t1, "explore the site", site)
    assert r1["kind"] == "explore" and r1["states"] >= 5 and ops.get_task(t1)["status"] == "done"
    assert r1["navigators"] == settings.navigators_per_site == 2     # default: two agents on the one site graph
    assert r1["agents"] == ["navigator-desktop-1", "navigator-desktop-2"]
    t2 = ops.create_task("chat", "take me to the cart")
    r2 = o.run_task(t2, "take me to the cart", site)
    assert r2["kind"] == "automate" and r2["status"] == "done"
    recs = o.recommend(site)
    assert {"next_actions", "frontier", "suggested_workflows", "demo_candidates"} <= set(recs)


@pytest.mark.integration
@pytest.mark.browser
@pytest.mark.parametrize("explicit, named_in_request, expected", [(1, None, 1), (None, 3, 3), (99, None, 16)])
def test_number_of_navigators_is_configurable(store, settings, site_url, explicit, named_in_request, expected):
    # Only the agent count is checked here, so keep each exploration tiny.
    settings.navigators_per_site = 2
    extra = {"navigators": named_in_request} if named_in_request else None
    o = Orchestrator(settings, store, HashEmbedder(), _llm(extra), BlobStore(settings.blob_dir), NullEvents())
    site = o.register_site(site_url + "/about.html")        # one small page: nothing much to explore
    ops = Ops(store, settings.uid, site.domain)
    t = ops.create_task("chat", "explore the site")
    res = o.run_task(t, "explore the site", site, navigators=explicit)
    assert res["kind"] == "explore", res
    assert res["navigators"] == expected and len(res["agents"]) == expected
