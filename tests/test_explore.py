import pytest
from sitegraph.agents.navigator import Navigator
from sitegraph.browser.session import BrowserSession
from sitegraph.db import device_key
from sitegraph.embed import HashEmbedder
from sitegraph.normalize import domain_of
from sitegraph.policy import ExplorationPolicy
from sitegraph.schema import ensure_device_schema

pytestmark = [pytest.mark.integration, pytest.mark.browser]


@pytest.fixture
def explored(store, settings, site_url):
    domain = domain_of(site_url)
    key = device_key(settings.uid, domain, "desktop")
    ensure_device_schema(store, key, 384)
    with BrowserSession() as s:
        nav = Navigator(store, key, domain, s, settings, HashEmbedder())
        report = nav.explore(site_url + "/index.html", ExplorationPolicy(max_depth=2))
    return key, report


def test_explore_builds_states_actions_and_transitions(store, explored):
    key, report = explored
    assert len(report.states) >= 6 and report.actions >= 5 and report.transitions >= 1
    assert store.ro_query(key, "MATCH (s:State:Entry) RETURN count(s)") == [[1]]
    assert store.ro_query(key, "MATCH (:Component {kind:'menu'})-[:HAS_CSTATE]->(a)-[:TRANSITION]->(b) "
                               "RETURN a.name, b.name LIMIT 1") == [["absent", "open"]]
    assert store.ro_query(key, "MATCH (e:Element {is_template:true}) RETURN max(e.count)") == [[6]]
    assert store.ro_query(key, "MATCH (s:State:Blocking) RETURN count(s) > 0") == [[True]]
    assert store.ro_query(key, "MATCH ()-[r:ACTION]->() WHERE size(r.pre_cstates) > 0 RETURN count(r) > 0") == [[True]]


def test_explore_never_clicks_risky(store, explored):
    key, report = explored
    assert report.blocked >= 1
    rows = store.ro_query(key, "MATCH (e:Element) WHERE e.risk_cost > 1 RETURN e.name_sample, e.tried")
    assert rows and all(tried is False for _, tried in rows)
    assert store.ro_query(key, "MATCH (e:Element)<-[:CONTAINS]-() WHERE e.risk_cost > 1 "
                               "MATCH ()-[r:ACTION]->() WHERE r.el_gid = e.gid RETURN count(r)") == [[0]]


def test_explore_with_forms_records_required_inputs(store, settings, site_url):
    domain = domain_of(site_url)
    key = device_key(settings.uid, domain, "desktop") + ":forms"
    ensure_device_schema(store, key, 384)
    with BrowserSession() as s:
        Navigator(store, key, domain, s, settings, HashEmbedder()).explore(
            site_url + "/index.html", ExplorationPolicy(max_depth=1, allow_forms=True))
    rows = store.ro_query(key, "MATCH ()-[r:ACTION]->() WHERE size(r.requires_inputs) > 0 RETURN r.value_var")
    assert rows and rows[0][0] == "q"
    store.delete(key)
