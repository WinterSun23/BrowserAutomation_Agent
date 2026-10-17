import json

import pytest
from sitegraph.agents.curator import Curator
from sitegraph.agents.navigator import Navigator
from sitegraph.browser.session import BrowserSession
from sitegraph.db import device_key
from sitegraph.embed import HashEmbedder
from sitegraph.llm import FakeLLM, OfflineLLM
from sitegraph.normalize import domain_of
from sitegraph.policy import ExplorationPolicy
from sitegraph.schema import ensure_device_schema

pytestmark = [pytest.mark.integration, pytest.mark.browser]


@pytest.fixture(scope="module")
def shop(store, msettings, site_url):
    """Explore the category list once (no LLM at all) and annotate it with the offline defaults."""
    domain = domain_of(site_url)
    key = device_key(msettings.uid, domain, "desktop")
    ensure_device_schema(store, key, 384)
    with BrowserSession() as s:
        Navigator(store, key, domain, s, msettings, HashEmbedder(), OfflineLLM()).explore(
            site_url + "/categories.html", ExplorationPolicy(max_depth=1))
    Curator(store, key, msettings, HashEmbedder(), OfflineLLM()).run_all(1)
    return store, msettings, key, domain, site_url


def _nav(shop, session, llm=None):
    store, settings, key, domain, _ = shop
    return Navigator(store, key, domain, session, settings, HashEmbedder(), llm or OfflineLLM())


def _choices(store, key):
    return [json.loads(r[0]) for r in store.ro_query(key, "MATCH (o:Observation {kind:'item_choice'}) RETURN o.detail")]


def test_the_shared_category_page_is_one_state_with_example_titles(shop):
    store, _, key, _, _ = shop
    assert store.ro_query(key, "MATCH (p:Page {url_template:'/category.html?name'})-[:HAS_STATE]->(s:State) "
                               "RETURN count(s)") == [[1]]                      # one node for every category
    titles = store.ro_query(key, "MATCH (p:Page {url_template:'/category.html?name'}) RETURN p.example_titles")[0][0]
    assert "Category: Travel" in titles                                         # the instance the explorer clicked


def test_offline_request_for_a_named_item_reaches_that_item_by_text(shop):
    store, _, key, _, _ = shop
    with BrowserSession() as s:
        res = _nav(shop, s).execute_goal("open the mystery category")           # no LLM, no classifier item
        assert res.status == "done", res.explanation
        assert s.url.endswith("/category.html?name=mystery")                    # not the entry page, not the first item
    assert {"item": "Mystery", "method": "text", "candidates": 6} in _choices(store, key)


def test_example_titles_accumulate_and_feed_later_matching(shop):
    store, _, key, domain, _ = shop
    titles =store.ro_query(key, "MATCH (p:Page {url_template:'/category.html?name'}) RETURN p.example_titles")[0][0]
    assert {"Category: Travel", "Category: Mystery"} <= set(titles)             # the executed visit was recorded too
    Curator(store, key, shop[1], HashEmbedder(), OfflineLLM()).annotate()      # refresh picks the new example up
    assert store.ro_query(key, "MATCH (i:Intent {text:'open Category: Mystery'})-[:FULFILLED_BY]->(s:State) "
                               "RETURN count(s)") == [[1]]
    assert store.ro_query(key, "MATCH (p:Page {url_template:'/category.html?name'})-[:HAS_STATE]->(s:State) "
                               "RETURN s.examples_n >= 2") == [[True]]


def test_an_llm_picks_the_item_when_the_request_does_not_name_it(shop):
    store, _, key, _, _ = shop

    def responder(system, user):
        # the list is shown as '0. Travel', '1. Mystery', ...; choose the one about whodunits
        idx = next(int(l.split(".")[0]) for l in user.splitlines() if l.strip().endswith("Mystery"))
        return json.dumps({"index": idx})

    with BrowserSession() as s:
        res = _nav(shop, s, FakeLLM(responder)).execute_goal("open the category about whodunits")
        assert res.status == "done", res.explanation
        assert s.url.endswith("/category.html?name=mystery")
    assert any(c["method"] == "llm" and c["item"] == "Mystery" for c in _choices(store, key))


def test_an_item_name_alone_finds_the_page_type_it_leads_to(shop):
    """'Classics' was never visited as a page and the request does not say 'category': the target screen is found
    because 'Classics' is one of the list items whose click leads to the shared category state."""
    store, _, key, _, _ = shop
    with BrowserSession() as s:
        res = _nav(shop, s).execute_goal("take me to classics")
        assert res.status == "done", res.explanation
        assert s.url.endswith("/category.html?name=classics")


def test_without_any_hint_the_first_item_is_used(shop):
    with BrowserSession() as s:
        res = _nav(shop, s).execute_goal("open a category")                    # nothing to choose by, offline
        assert res.status == "done", res.explanation
        assert s.url.endswith("/category.html?name=travel")


def test_find_screens_names_the_list_item_and_how_to_reach_it(shop):
    """The shared category state is named after whichever category was seen last; find_screens must not let that
    name hide the entry the request asked for (an agent read 'Poetry' and concluded Travel did not exist)."""
    from sitegraph.tools import ToolContext, build_toolbox

    store, settings, key, domain, _ = shop
    ctx = ToolContext(settings=settings, store=store, key=key, domain=domain, embedder=HashEmbedder(), llm=OfflineLLM())
    hits = build_toolbox(ctx).call("find_screens", {"query": "take me to the travel category"})
    top = hits.data[0]
    assert top["item"] == "Travel"
    assert "'Travel'" in top["description"] and "run_goal" in top["description"] and "item" in top["description"]
    assert all("item" not in h for h in hits.data[1:] if h["gid"] != top["gid"])   # ordinary hits stay as they were
