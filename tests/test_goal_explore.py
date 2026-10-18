import json
from types import SimpleNamespace

import pytest
from sitegraph.agents.goal_explore import goal_priority
from sitegraph.agents.orchestrator import Orchestrator
from sitegraph.blobs import BlobStore
from sitegraph.embed import HashEmbedder
from sitegraph.events import NullEvents
from sitegraph.llm import FakeLLM
from sitegraph.ops import Ops
from tests.helpers import site_responder


def _el(name, href=None):
    return SimpleNamespace(name_sample=name, name_norm=name.lower(), href_template=href)


def test_elements_that_resemble_the_goal_get_a_higher_priority():
    prio = goal_priority(HashEmbedder(), "open the shopping cart")
    cart, about, empty = prio(_el("Shopping cart", "/cart.html")), prio(_el("About us", "/about.html")), prio(_el(""))
    assert cart > about >= 0.0 and empty == 0.0


def _llm(goal="open cart"):
    def r(system, user):
        if "Classify" in user:
            return json.dumps({"kind": "automate", "goal": goal, "params": {}})
        return site_responder(system, user)
    return FakeLLM(r)


def _orch(store, settings, goal="open cart", **cfg):
    settings.min_target_score = 0.3
    settings.navigators_per_site, settings.dynamic_spawn = 1, False
    for k, v in cfg.items():
        setattr(settings, k, v)
    return Orchestrator(settings, store, HashEmbedder(), _llm(goal), BlobStore(settings.blob_dir), NullEvents())


@pytest.mark.integration
@pytest.mark.browser
def test_a_request_on_a_never_explored_site_explores_toward_the_goal_then_runs(store, settings, site_url):
    o = _orch(store, settings, goal_explore_round_states=4)
    site = o.register_site(site_url + "/index.html")
    ops = Ops(store, settings.uid, site.domain)
    res = o.run_task(ops.create_task("chat", "take me to the cart"), "take me to the cart", site)
    assert res["kind"] == "automate" and res["status"] == "done", res
    info = res["goal_explore"]
    assert info["found"] is True and info["rounds"] >= 1 and info["states_added"] >= 1
    assert store.ro_query(site.key, "MATCH (s:State) RETURN count(s)")[0][0] >= 2


@pytest.mark.integration
@pytest.mark.browser
def test_exploration_toward_an_unmatchable_goal_stops_within_the_budget(store, settings, site_url):
    o = _orch(store, settings, "zzqx vvkk", goal_explore_round_states=3, goal_explore_max_rounds=2,
              min_target_score=0.9)
    site = o.register_site(site_url + "/index.html")
    ops = Ops(store, settings.uid, site.domain)
    res = o.run_task(ops.create_task("chat", "zzqx vvkk"), "zzqx vvkk", site)
    info = res["goal_explore"]
    assert info["found"] is False and 1 <= info["rounds"] <= 2 and info["reason"]
    assert res["status"] == "no_target"


@pytest.mark.integration
@pytest.mark.browser
def test_nothing_is_explored_when_the_map_already_answers(store, settings, site_url):
    o = _orch(store, settings)
    site = o.register_site(site_url + "/index.html")
    ops = Ops(store, settings.uid, site.domain)
    o.run_task(ops.create_task("chat", "explore the site"), "explore the site", site, kind="explore")
    res = o.run_task(ops.create_task("chat", "take me to the cart"), "take me to the cart", site)
    assert res["status"] == "done" and "goal_explore" not in res
    off = _orch(store, settings, "zzqx vvkk", goal_explore=False)
    r2 = off.run_task(ops.create_task("chat", "zzqx vvkk"), "zzqx vvkk", site)
    assert "goal_explore" not in r2
