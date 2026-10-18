import re
from pathlib import Path
import pytest
from sitegraph.agents.curator import Curator
from sitegraph.agents.navigator import Navigator
from sitegraph.agents.workflows import load_workflow_steps
from sitegraph.blobs import BlobStore
from sitegraph.browser.session import BrowserSession
from sitegraph.catalog import REL_CATALOG
from sitegraph.config import Settings
from sitegraph.versions import ALGO_VERSION
from sitegraph.db import device_key
from sitegraph.embed import HashEmbedder
from sitegraph.graph.map_writer import MapWriter
from sitegraph.graph.queries import best_routes, entry_state
from sitegraph.llm import FakeLLM
from sitegraph.migrate import rehash
from sitegraph.normalize import domain_of
from sitegraph.policy import ExplorationPolicy
from sitegraph.schema import ensure_device_schema
from tests.helpers import site_responder

pytestmark = [pytest.mark.integration, pytest.mark.browser]
SRC = Path(__file__).parents[1] / "src" / "sitegraph"


def test_catalog_types_are_used_in_code():
    code = "\n".join(p.read_text(encoding="utf-8") for p in SRC.rglob("*.py") if p.name != "catalog.py")
    for t in REL_CATALOG:
        assert re.search(rf"\b{t}\b", code), f"catalogued type {t} is not used anywhere"


@pytest.fixture(scope="module")
def built(store, msettings, site_url, tmp_path_factory):
    domain = domain_of(site_url)
    key = device_key(msettings.uid, domain, "desktop")
    ensure_device_schema(store, key, 384)
    blobs = BlobStore(tmp_path_factory.mktemp("blobs"))
    MapWriter(store, key, msettings).ensure_site(domain, "desktop", "Desktop Chrome", "hash", 384)
    with BrowserSession() as s:
        nav = Navigator(store, key, domain, s, msettings, HashEmbedder(), blobs=blobs)
        nav.explore(site_url + "/index.html", ExplorationPolicy(max_depth=2))
        Curator(store, key, msettings, HashEmbedder(), FakeLLM(site_responder)).run_all(1)
        s.goto(site_url + "/index.html")         # exploring may end on the cart page: "open cart" would record no step
        res = nav.execute_goal("open cart")
    from sitegraph.agents.workflows import save_workflow_from_run
    wf = save_workflow_from_run(store, key, HashEmbedder(), res.run_gid, "cart", "open my shopping cart")
    return key, domain, blobs, wf


def test_graph_uses_only_catalogued_types(store, built):
    key = built[0]
    types = {r[0] for r in store.ro_query(key, "CALL db.relationshipTypes()")}
    assert types <= set(REL_CATALOG), f"undocumented relationship types: {types - set(REL_CATALOG)}"


def test_site_node_versions(store, built):
    key = built[0]
    assert store.ro_query(key, "MATCH (s:Site) RETURN s.schema_version, s.algo_version, s.embed_dim") == [[2, ALGO_VERSION, 384]]


def test_a_different_embedding_model_is_refused(store, built):
    key, domain, _, _ = built
    with pytest.raises(ValueError, match="hash"):
        MapWriter(store, key, Settings()).ensure_site(domain, "desktop", "Desktop Chrome", "BAAI/bge-base-en-v1.5", 384)


def test_rehash_is_lossless_when_algorithm_unchanged(store, built, msettings):
    key, domain, blobs, wf = built
    store.query(key, "MATCH ()-[r:ACTION]->() WITH r LIMIT 1 MATCH (x:Element) WITH r, x LIMIT 1 "
                     "SET r.pre_actions = ['click|' + x.gid]")                   # a learned step (agent-layer A5)
    before = store.ro_query(key, "MATCH (s:State) WITH count(s) AS s MATCH ()-[a:ACTION]->() RETURN s, count(a)")[0]
    learned = store.ro_query(key, "MATCH ()-[r:ACTION]->() WHERE size(coalesce(r.pre_actions, [])) > 0 "
                                  "RETURN r.pre_actions")
    report = rehash(store, key, blobs, msettings, domain)
    after = store.ro_query(key, "MATCH (s:State) WITH count(s) AS s MATCH ()-[a:ACTION]->() RETURN s, count(a)")[0]
    assert after == before and report["states"] == before[0]
    assert report["dropped_actions"] == 0 and report["dropped_transitions"] == 0
    assert load_workflow_steps(store, key, wf)
    entry = entry_state(store, key)
    cart = store.ro_query(key, "MATCH (:Page {url_template:'/cart.html'})-[:HAS_STATE]->(s:State) RETURN s.gid LIMIT 1")[0][0]
    assert best_routes(store, key, entry[0], cart, 1)
    kept = store.ro_query(key, "MATCH ()-[r:ACTION]->() WHERE size(coalesce(r.pre_actions, [])) > 0 RETURN r.pre_actions")
    assert kept == learned                                                     # learned steps survive the migration
    for (token,) in [(t,) for row in kept for t in row[0]]:
        assert store.ro_query(key, "MATCH (x:Element {gid:$g}) RETURN count(x)", {"g": token.split("|")[1]}) == [[1]]
