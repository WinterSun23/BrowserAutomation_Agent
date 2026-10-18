import json
import re

import pytest
from sitegraph.agents.frontier import FrontierGroup, choose_frontier, group_frontier
from sitegraph.agents.orchestrator import Orchestrator
from sitegraph.blobs import BlobStore
from sitegraph.db import device_key
from sitegraph.embed import HashEmbedder
from sitegraph.events import NullEvents
from sitegraph.llm import FakeLLM
from sitegraph.ops import Ops
from sitegraph.schema import ensure_device_schema
from tests.helpers import site_responder


def _word(i: int) -> str:
    s = ""
    for _ in range(3):
        s, i = s + "abcdefghijklmnopqrstuvwxyz"[i % 26], i // 26
    return s


def _elements(store, key, rows):
    """rows: (el_gid, name, href_template, component gid)"""
    store.query(key, "UNWIND $rows AS r MERGE (k:Component {gid:r[3]}) MERGE (cs:ComponentState {gid:r[3] + ':cs'}) "
                     "MERGE (k)-[:HAS_CSTATE]->(cs) CREATE (cs)-[:CONTAINS]->(:Element {gid:r[0], name_sample:r[1], "
                     "role:'link', href_template:r[2], is_template:false})", {"rows": [list(r) for r in rows]})


def test_groups_collapse_a_template_into_one_line(store, settings):
    key = device_key(settings.uid, "shop.test", "desktop")
    ensure_device_schema(store, key, settings.embed_dim)
    ops = Ops(store, settings.uid, "shop.test")
    _elements(store, key, [(f"el:{i}", f"Category {i}", "/category.html?name", "comp:cats") for i in range(50)])
    for i in range(50):
        ops.add_item("sc", key, "click", "http://shop.test/categories.html", 1, el_gid=f"el:{i}")
    groups = group_frontier(ops, "sc", store, key, 40)
    assert len(groups) == 1 and groups[0].count == 50 and len(groups[0].item_gids) == 50
    assert groups[0].sample.startswith("Category") and groups[0].page == "/categories.html"


def test_the_group_list_is_bounded(store, settings):
    key = device_key(settings.uid, "big.test", "desktop")
    ensure_device_schema(store, key, settings.embed_dim)
    ops = Ops(store, settings.uid, "big.test")
    store.query(ops.key, "UNWIND range(0, 1999) AS i CREATE (:WorkItem {gid:'wk:' + toString(i), scope:'big', "
                         "status:'open', kind:'page', url:'http://big.test/s' + $w[i] + '/x.html', depth:1, "
                         "priority:0.0, created:i})", {"w": [_word(i) for i in range(2000)]})
    groups = group_frontier(ops, "big", store, key, 40)
    assert len(groups) <= 40
    assert sum(g.count for g in groups) == 2000
    llm = FakeLLM(lambda s, u: "{}")
    choose_frontier(llm, groups, None, None, "")
    assert len(llm.calls[0][1]) < 8000                       # the prompt stays bounded


def test_next_item_respects_min_priority(store, settings):
    ops = Ops(store, settings.uid, "prio.test")
    ops.register_agent("a1", "explorer")
    low = ops.add_item("sp", "g", "page", "http://prio.test/low", 0, priority=0)
    high = ops.add_item("sp", "g", "page", "http://prio.test/high", 0, priority=100)
    item, _ = ops.next_item("sp", "a1", "g", min_priority=1)
    assert item["gid"] == high
    ops.finish_item(high, "a1")
    assert ops.next_item("sp", "a1", "g", min_priority=1) == (None, False)
    item, _ = ops.next_item("sp", "a1", "g")
    assert item["gid"] == low


def test_junk_or_unknown_ids_give_an_invalid_choice():
    groups = [FrontierGroup("g1", "click", "Products", 1, 0, "/index.html", ["w1"])]
    bad = choose_frontier(FakeLLM(lambda s, u: "no json"), groups, None, "goal", "")
    assert bad.ok is False and bad.pick == [] and bad.stop is False
    odd = choose_frontier(FakeLLM(lambda s, u: json.dumps({"pick": ["g9", "g1"], "skip": [{"pattern": "("}]})),
                          groups, None, None, "")
    assert odd.ok and odd.pick == ["g1"] and odd.skip == []


def _group_id(user: str, sample: str) -> str | None:
    m = re.search(r"^(g\d+) \| [a-z]+ \| '" + re.escape(sample), user, re.M)
    return m.group(1) if m else None


def _guided_llm(decide):
    """decide(call_no, user) -> dict answer for the frontier prompt; everything else as in other tests."""
    calls = []

    def r(system, user):
        if "frontier" in system.lower():
            calls.append(user)
            return json.dumps(decide(len(calls), user)) if not isinstance(decide, str) else decide
        return site_responder(system, user)
    return FakeLLM(r), calls


def _setup(store, settings, site_url, decide, **cfg):
    settings.navigators_per_site, settings.dynamic_spawn, settings.min_target_score = 1, False, 0.3
    for k, v in cfg.items():
        setattr(settings, k, v)
    llm, calls = _guided_llm(decide)
    o = Orchestrator(settings, store, HashEmbedder(), llm, BlobStore(settings.blob_dir), NullEvents())
    site = o.register_site(site_url + "/index.html")
    ops = Ops(store, settings.uid, site.domain)
    return o, site, ops, calls


def _templates(store, site):
    return {r[0] for r in store.ro_query(site.key, "MATCH (p:Page) RETURN p.url_template")}


def _pick_products(n, user):
    g = _group_id(user, "Products")
    return {"pick": [g] if g else [], "skip": [], "stop": False}


@pytest.mark.integration
@pytest.mark.browser
def test_a_guided_round_explores_only_what_the_model_picked(store, settings, site_url):
    o, site, ops, calls = _setup(store, settings, site_url, _pick_products, guided_max_rounds=1)
    policy = o.profile.policy_for(site.domain, site.device)
    res = o._explore_guided(site, policy, ops.create_task("explore", "x"), 1)
    assert res["mode"] == "guided" and res["rounds"] == 1, {k: v for k, v in res.items() if k != "per_agent"}
    assert "'Products'" in calls[0] and "'About'" in calls[0]
    tpls = _templates(store, site)
    assert "/products.html" in tpls and "/about.html" not in tpls


@pytest.mark.integration
@pytest.mark.browser
def test_skip_rules_close_items_and_are_saved(store, settings, site_url):
    def decide(n, user):
        return {**_pick_products(n, user), "skip": [{"pattern": "contact|about", "reason": "not useful"}]}

    o, site, ops, calls = _setup(store, settings, site_url, decide, guided_max_rounds=1)
    policy = o.profile.policy_for(site.domain, site.device)
    res = o._explore_guided(site, policy, ops.create_task("explore", "x"), 1)
    assert res["skipped_groups"] >= 1
    rows = store.ro_query(ops.key, "MATCH (w:WorkItem {status:'done', error:'skipped by planner'}) RETURN w.el_gid")
    assert rows, "the Contact item should be closed"
    assert "contact|about" in o.profile.policy_for(site.domain, site.device).deny_patterns


@pytest.mark.integration
@pytest.mark.browser
def test_stop_ends_the_exploration(store, settings, site_url):
    def decide(n, user):
        return _pick_products(n, user) if n == 1 else {"pick": [], "skip": [], "stop": True}

    o, site, ops, calls = _setup(store, settings, site_url, decide, guided_max_rounds=5)
    res = o._explore_guided(site, o.profile.policy_for(site.domain, site.device), ops.create_task("explore", "x"), 1)
    assert res["rounds"] == 1 and len(calls) == 2 and "stop" in res["reason"]


@pytest.mark.integration
@pytest.mark.browser
def test_junk_choice_falls_back_to_priority_order(store, settings, site_url):
    o, site, ops, calls = _setup(store, settings, site_url, "certainly! here is my choice", guided_max_rounds=1)
    res = o._explore_guided(site, o.profile.policy_for(site.domain, site.device), ops.create_task("explore", "x"), 1)
    assert res["rounds"] == 1 and res["states"] >= 2 and len(calls) == 1
    assert "/products.html" in _templates(store, site)
