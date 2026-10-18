import json

import pytest
from sitegraph.agents.bootstrap import SiteProfile
from sitegraph.agents.orchestrator import Orchestrator, decide_mode, mode_decision
from sitegraph.blobs import BlobStore
from sitegraph.cli import build_parser
from sitegraph.config import Settings
from sitegraph.embed import HashEmbedder
from sitegraph.events import NullEvents
from sitegraph.llm import FakeLLM, OfflineLLM
from sitegraph.ops import Ops
from tests.helpers import site_responder


def _p(size="medium", sitemap=None):
    return SiteProfile("a shop", [], size, [], [], sitemap)


def test_decide_mode_rules():
    s = Settings()
    assert decide_mode(_p("large"), False, None, s) == "brute"                 # no model, whatever the profile
    assert "no model" in mode_decision(None, False, None, s)[1]
    assert decide_mode(_p("large"), True, None, s) == "guided"
    assert decide_mode(_p("medium", 500), True, None, s) == "guided"           # sitemap > auto_sitemap_large
    assert decide_mode(_p("small"), True, None, s) == "brute"
    assert decide_mode(None, True, None, s) == "probe"                         # undecided: probe first
    assert decide_mode(_p("medium"), True, 30, s) == "guided"                  # probe left a big frontier
    assert decide_mode(None, True, 3, s) == "brute"


def _orch(store, settings, llm, **cfg):
    settings.navigators_per_site, settings.dynamic_spawn, settings.min_target_score = 1, False, 0.3
    for k, v in cfg.items():
        setattr(settings, k, v)
    o = Orchestrator(settings, store, HashEmbedder(), llm, BlobStore(settings.blob_dir), NullEvents())
    site = o.register_site(site_url_holder[0] + "/index.html")
    return o, site, Ops(store, settings.uid, site.domain)


site_url_holder: list[str] = []


@pytest.fixture(autouse=True)
def _url(site_url):
    site_url_holder[:] = [site_url]


@pytest.mark.integration
@pytest.mark.browser
def test_auto_without_a_model_is_brute(store, settings):
    o, site, ops = _orch(store, settings, OfflineLLM())
    res = o.run_task(ops.create_task("chat", "explore the site"), "explore the site", site)
    assert res["kind"] == "explore" and res["mode"] == "brute" and "no model" in res["mode_reason"], res
    assert res["states"] >= 5


def _frontier_llm(calls):
    def r(system, user):
        if "frontier" in system.lower():
            calls.append(user)
            ids = [ln.split(" | ")[0] for ln in user.splitlines() if ln.startswith("g") and " | " in ln]
            return json.dumps({"pick": ids[:2], "skip": [], "stop": False})
        if "Classify" in user:
            return json.dumps({"kind": "explore"})
        return site_responder(system, user)
    return FakeLLM(r)


@pytest.mark.integration
@pytest.mark.browser
def test_auto_switches_to_guided_after_the_probe(store, settings):
    calls = []
    o, site, ops = _orch(store, settings, _frontier_llm(calls), auto_frontier_large=2, auto_probe_states=2,
                         guided_max_rounds=2)
    res = o.run_task(ops.create_task("chat", "explore the site"), "explore the site", site)
    assert res["kind"] == "explore" and res["mode"] == "guided", res
    assert res["probe_groups"] > 2 and "probe" in res["mode_reason"] and res["rounds"] >= 1 and calls


@pytest.mark.integration
@pytest.mark.browser
def test_auto_stays_brute_on_a_small_frontier(store, settings):
    calls = []
    o, site, ops = _orch(store, settings, _frontier_llm(calls), auto_frontier_large=1000, auto_probe_states=2)
    res = o.run_task(ops.create_task("chat", "explore the site"), "explore the site", site)
    assert res["mode"] == "brute" and not calls and res["states"] >= 5, res


def test_the_mode_flag_overrides_config():
    args = build_parser().parse_args(["run", "https://example.com/", "explore the site", "--mode", "brute"])
    assert args.mode == "brute"
    assert build_parser().parse_args(["run", "https://example.com/", "x"]).mode is None   # config decides


@pytest.mark.integration
@pytest.mark.browser
def test_the_mode_argument_overrides_the_setting(store, settings):
    calls = []
    o, site, ops = _orch(store, settings, _frontier_llm(calls), exploration_mode="guided")
    res = o.explore(site, o.profile.policy_for(site.domain, site.device), ops.create_task("x", "x"), 1, mode="brute")
    assert res["mode"] == "brute" and "asked for" in res["mode_reason"] and not calls
    with pytest.raises(ValueError):
        o.explore(site, o.profile.policy_for(site.domain, site.device), "t", 1, mode="random")


@pytest.mark.integration
@pytest.mark.browser
def test_goal_search_uses_guided_rounds_with_a_model(store, settings):
    calls = []
    o, site, ops = _orch(store, settings, _frontier_llm(calls), goal_explore_round_states=3,
                         goal_explore_max_rounds=2)
    res = o.explore_toward(site, "red hoodie details", "show me the red hoodie", o.profile.policy_for(
        site.domain, site.device), ops.create_task("x", "x"), 1)
    assert res["mode"] == "guided" and res["rounds"] >= 1, res
    assert calls and "red hoodie details" in calls[0]
