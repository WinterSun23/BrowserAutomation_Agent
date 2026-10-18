import json
from types import SimpleNamespace

import pytest
from sitegraph.agents import goal_explore
from sitegraph.agents.curator import Curator
from sitegraph.agents.execution_agent import ExecutionAgent
from sitegraph.agents.goal_explore import found_target, goal_priority, goal_words, quick_match
from sitegraph.agents.orchestrator import Orchestrator, SiteInfo
from sitegraph.blobs import BlobStore
from sitegraph.config import Settings
from sitegraph.db import device_key, ops_key
from sitegraph.embed import HashEmbedder
from sitegraph.events import NullEvents
from sitegraph.graph.queries import Target
from sitegraph.llm import FakeLLM, OfflineLLM
from sitegraph.ops import Ops
from sitegraph.schema import ensure_device_schema
from tests.helpers import site_responder


def test_goal_words_drop_filler():
    assert goal_words("Take me to the price of the Red Hoodie, please") == {"price", "red", "hoodie"}
    assert quick_match("red hoodie", "Red Hoodie", "/product.html?id=2", "", 0.6)
    assert not quick_match("red hoodie", "Products", "/products.html", "Blue T-shirt", 0.6)
    assert not quick_match("show me", "anything", "", "", 0.1)                  # no content words: never a match


def test_a_weak_match_is_not_found_without_the_goal_words_on_the_page(monkeypatch):
    s = Settings()
    hits = []
    monkeypatch.setattr(goal_explore, "find_targets", lambda *a, **k: hits)
    hits[:] = [Target("State", "s1", "Some page", 0.4)]
    assert found_target(None, "k", HashEmbedder(), "red hoodie", None, s, {"s1": "nothing relevant"}) is None
    t = found_target(None, "k", HashEmbedder(), "red hoodie", None, s, {"s1": "the Red Hoodie, $30"})
    assert t is not None and t.gid == "s1"
    hits[:] = [Target("State", "s1", "Some page", 0.7)]                       # strong on its own
    assert found_target(None, "k", HashEmbedder(), "red hoodie", None, s, {}) is not None
    hits[:] = [Target("State", "s1", "Red hoodie", 0.2)]                      # below min_target_score: never
    assert found_target(None, "k", HashEmbedder(), "red hoodie", None, s, {"s1": "red hoodie"}) is None


def _el(name):
    return SimpleNamespace(name_sample=name, name_norm=name.lower(), href_template=None)


def test_pagination_gets_a_bonus_for_unseen_items():
    plain = goal_priority(HashEmbedder(), "book Logan Kade")
    bonus = goal_priority(HashEmbedder(), "book Logan Kade", unseen_item=True)
    assert bonus(_el("Next")) == pytest.approx(plain(_el("Next")) + 0.2)
    assert bonus(_el("Next")) > bonus(_el("About us"))
    assert bonus(_el("»")) > plain(_el("»")) and bonus(_el("About us")) == plain(_el("About us"))


def test_chat_hint_points_to_explore_toward(store, settings):
    key = device_key(settings.uid, "empty.test", "desktop")
    ensure_device_schema(store, key, settings.embed_dim)
    site = SiteInfo("empty.test", "desktop", key, ops_key(settings.uid, "empty.test"), "http://empty.test/")
    agent = ExecutionAgent(settings, store, HashEmbedder(), OfflineLLM(), None, site)
    hint = agent.map_hint("price of the red hoodie")
    assert "explore_toward" in hint and hint.index("explore_toward") < hint.index("by hand")


def _frontier_llm():
    def r(system, user):
        if "frontier" in system.lower():
            ids = [ln.split(" | ")[0] for ln in user.splitlines() if ln.startswith("g") and " | " in ln]
            return json.dumps({"pick": ids[:3], "skip": [], "stop": False})
        return site_responder(system, user)
    return FakeLLM(r)


def _orch(store, settings, site_url, llm, **cfg):
    settings.navigators_per_site, settings.dynamic_spawn = 1, False
    for k, v in cfg.items():
        setattr(settings, k, v)
    o = Orchestrator(settings, store, HashEmbedder(), llm, BlobStore(settings.blob_dir), NullEvents())
    site = o.register_site(site_url + "/index.html")
    return o, site, Ops(store, settings.uid, site.domain)


NEVER = dict(goal_found_score=0.99, min_target_score=0.99, goal_explore_round_states=2)


@pytest.mark.integration
@pytest.mark.browser
def test_rounds_share_one_queue(store, settings, site_url):
    o, site, ops = _orch(store, settings, site_url, OfflineLLM(), goal_explore_max_rounds=3, **NEVER)
    res = o.explore_toward(site, "zzqx vvkk", None, o.profile.policy_for(site.domain, site.device), "t", 1)
    assert res["found"] is False and res["rounds"] >= 2, res
    rows = ops.q("MATCH (w:WorkItem {scope:$s, kind:'page'}) WHERE w.url = $u RETURN w.attempts",
                 s=res["scope"], u=site.entry_url)
    assert [r[0] for r in rows] == [1]                    # the entry page was loaded once for all rounds


@pytest.mark.integration
@pytest.mark.browser
@pytest.mark.parametrize("with_model", [False, True])
def test_full_curation_runs_once(store, settings, site_url, monkeypatch, with_model):
    calls = {"run_all": 0, "annotate": 0}
    real_annotate = Curator.annotate

    def run_all(self, *a, **k):
        calls["run_all"] += 1
        return {}

    def annotate(self, *a, **k):
        calls["annotate"] += 1
        return real_annotate(self, *a, **k)

    monkeypatch.setattr(Curator, "run_all", run_all)
    monkeypatch.setattr(Curator, "annotate", annotate)
    o, site, ops = _orch(store, settings, site_url, _frontier_llm() if with_model else OfflineLLM(),
                         goal_explore_max_rounds=2, **NEVER)
    res = o.explore_toward(site, "zzqx vvkk", None, o.profile.policy_for(site.domain, site.device), "t", 1)
    assert res["mode"] == ("guided" if with_model else "brute") and res["rounds"] >= 1, res
    # one annotate per round, plus one for the inputs-first step when it set a control (the search box here)
    assert calls["run_all"] == 1 and calls["annotate"] == res["rounds"] + (1 if res["inputs_used"] else 0)


@pytest.mark.integration
@pytest.mark.browser
def test_next_round_starts_from_the_best_new_screens(store, settings, site_url):
    o, site, ops = _orch(store, settings, site_url, OfflineLLM(), goal_explore_max_rounds=2, **NEVER)
    res = o.explore_toward(site, "all products", None, o.profile.policy_for(site.domain, site.device), "t", 1)
    assert res["rounds"] == 2 and res["round_seeds"], res
    assert any(u.endswith("/products.html") for u in res["round_seeds"][0])
    rows = ops.q("MATCH (w:WorkItem {scope:$s, kind:'page'}) WHERE w.url ENDS WITH '/products.html' "
                 "RETURN w.depth, w.status", s=res["scope"])
    assert rows and rows[0][0] == 0                       # re-seeded with its full depth budget


@pytest.mark.integration
@pytest.mark.browser
def test_inputs_first_reaches_the_target_without_rounds(store, settings, site_url):
    o, site, ops = _orch(store, settings, site_url, OfflineLLM(), min_target_score=0.3)
    res = o.explore_toward(site, "search results for widget", None, o.profile.policy_for(site.domain, site.device),
                           "t", 1)
    assert res["found"] is True and res["how"] == "inputs" and res["rounds"] == 0, res
    assert res["inputs_used"] == ["q"]
    texts = store.ro_query(site.key, "MATCH (p:Page) RETURN p.example_urls, p.title")
    assert "widget" not in json.dumps(texts)                       # values are never stored
