import json
import pytest
from sitegraph.agents.curator import Curator
from sitegraph.agents.navigator import Navigator
from sitegraph.agents.workflows import save_workflow_from_run
from sitegraph.browser.session import BrowserSession
from sitegraph.db import device_key
from sitegraph.embed import HashEmbedder
from sitegraph.llm import FakeLLM
from sitegraph.normalize import domain_of
from sitegraph.policy import ExplorationPolicy
from sitegraph.profile import Profile
from sitegraph.schema import ensure_device_schema
from sitegraph.semantic.distiller import Distiller, semantic_key
from tests.helpers import site_responder

pytestmark = [pytest.mark.integration, pytest.mark.browser]


def responder(system, user):
    if "UI pattern" in user:
        return json.dumps({"name": "pattern"})
    if "role of this screen" in user:
        return json.dumps({"role": "screen"})
    if "Name this flow" in user:
        return json.dumps({"name": "go to cart"})
    return site_responder(system, user)


@pytest.fixture(scope="module")
def two_devices(store, msettings, site_url):
    domain = domain_of(site_url)
    prof = Profile(store, msettings.uid, HashEmbedder())
    keys = {}
    for dev in ("desktop", "phone"):
        key = device_key(msettings.uid, domain, dev)
        ensure_device_schema(store, key, 384)
        prof.register_site(domain, dev, key, "o", site_url + "/index.html")
        with BrowserSession(dev) as s:
            nav = Navigator(store, key, domain, s, msettings, HashEmbedder())
            nav.explore(site_url + "/index.html", ExplorationPolicy(max_depth=1))
            Curator(store, key, msettings, HashEmbedder(), FakeLLM(responder)).run_all(1)
            if dev == "desktop":
                s.goto(site_url + "/index.html")     # exploring may end on the cart page: "open cart" would record no step
                res = nav.execute_goal("open cart")
                save_workflow_from_run(store, key, HashEmbedder(), res.run_gid, "cart", "open my shopping cart")
        keys[dev] = key
    return prof, keys, domain


def test_uipatterns_and_logical_states(store, msettings, two_devices):
    prof, keys, _ = two_devices
    d = Distiller(store, msettings.uid, HashEmbedder(), FakeLLM(responder), msettings, prof)
    stats = d.rebuild()
    sk = semantic_key(msettings.uid)
    assert stats["ui_patterns"] >= 3 and stats["logical_states"] >= 3 and stats["aligned"] >= 3
    assert store.ro_query(sk, "MATCH (l:LogicalState)-[r:REALISED_AS]->(:DeviceStateRef) "
                              "WITH l, collect(DISTINCT r.device) AS ds WHERE size(ds) = 2 RETURN count(l) >= 3") == [[True]]
    assert store.ro_query(sk, "MATCH (:ComponentRef)-[:INSTANCE_OF]->(u:UIPattern) RETURN count(DISTINCT u) >= 3") == [[True]]
    for k in keys.values():
        assert store.ro_query(k, "MATCH (n:LogicalState) RETURN count(n)") == [[0]]


def test_distiller_is_idempotent(store, msettings, two_devices):
    prof, _, _ = two_devices
    d = Distiller(store, msettings.uid, HashEmbedder(), FakeLLM(responder), msettings, prof)
    a = d.rebuild()
    b = d.rebuild()
    assert a == b
    sk = semantic_key(msettings.uid)
    assert store.ro_query(sk, "MATCH (r:DeviceStateRef) WHERE NOT (r)<-[:REALISED_AS]-() RETURN count(r)") == [[0]]
