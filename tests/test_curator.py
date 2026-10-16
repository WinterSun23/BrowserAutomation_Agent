import time
import pytest
from sitegraph.agents.curator import Curator
from sitegraph.agents.navigator import Navigator
from sitegraph.browser.session import BrowserSession
from sitegraph.db import device_key
from sitegraph.embed import HashEmbedder
from sitegraph.graph.activity import ActivityLog
from sitegraph.llm import FakeLLM
from sitegraph.normalize import domain_of
from sitegraph.policy import ExplorationPolicy
from sitegraph.schema import ensure_device_schema
from tests.helpers import site_responder

pytestmark = [pytest.mark.integration, pytest.mark.browser]


@pytest.fixture
def curated(store, settings, site_url):
    domain = domain_of(site_url)
    key = device_key(settings.uid, domain, "desktop")
    ensure_device_schema(store, key, 384)
    with BrowserSession() as s:
        Navigator(store, key, domain, s, settings, HashEmbedder()).explore(site_url + "/index.html",
                                                                           ExplorationPolicy(max_depth=2))
    cur = Curator(store, key, settings, HashEmbedder(), FakeLLM(site_responder))
    stats = cur.run_all(max_risk_cost=1)
    return key, cur, stats


def test_annotation_creates_summaries_and_intents(store, curated):
    key, _, stats = curated
    assert stats["annotated"] >= 5
    assert store.ro_query(key, "MATCH (s:State) WHERE s.summary IS NULL RETURN count(s)") == [[0]]
    assert store.ro_query(key, "MATCH (i:Intent {text:'open cart'})-[:FULFILLED_BY]->(:State) RETURN count(i) > 0") == [[True]]
    assert store.ro_query(key, "MATCH (e:Element) WHERE e.emb IS NULL RETURN count(e)") == [[0]]


def test_algorithm_scores_and_features(store, curated):
    key, _, stats = curated
    assert store.ro_query(key, "MATCH (s:State) WHERE s.pagerank IS NULL OR s.community IS NULL RETURN count(s)") == [[0]]
    assert stats["features"] >= 1
    assert store.ro_query(key, "MATCH (f:Feature) RETURN min(f.size) >= 1") == [[True]]


def test_retention_keeps_protected_runs(store, curated):
    key, cur, _ = curated
    log = ActivityLog(store, key)
    runs = []
    for _ in range(4):
        time.sleep(0.003)  # distinct start timestamps keep the retention order deterministic
        runs.append(log.start_run("execute", "n"))
    for r in runs:
        log.add_step(r, 0, "click", None, None, None, ok=True)
        log.finish_run(r, "done")
    store.query(key, "CREATE (:Shot {gid:'sh:x'})")
    store.query(key, "MATCH (sh:Shot {gid:'sh:x'}), (st:Step)-[:IN_RUN]->(:Run {gid:$r}) CREATE (sh)-[:FROM_STEP]->(st)",
                {"r": runs[0]})
    deleted = cur.apply_retention(keep=1)
    remaining = {r[0] for r in store.ro_query(key, "MATCH (r:Run) RETURN r.gid")}
    assert runs[0] in remaining and runs[-1] in remaining and deleted >= 2
