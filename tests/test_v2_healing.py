import json
import pytest
from sitegraph.browser.locate import ElementResolver
from sitegraph.browser.session import BrowserSession
from sitegraph.config import Settings
from sitegraph.graph.map_writer import MapWriter
from sitegraph.graph.queries import fulltext_elements
from sitegraph.llm import FakeLLM
from sitegraph.schema import ensure_device_schema

pytestmark = pytest.mark.integration


def test_fulltext_alternatives_and_locator_hits(store, gkey):
    ensure_device_schema(store, gkey, 384)
    store.query(gkey, """CREATE (:Element {gid:'el:a', role:'link', name_sample:'Shopping cart', locators:['css=#a'], fingerprint:'{}'}),
                                (:Element {gid:'el:b', role:'link', name_sample:'Cart', locators:['css=#b', 'css=#b2'], fingerprint:'{}'}),
                                (:Element {gid:'el:c', role:'button', name_sample:'Cart', locators:['css=#c'], fingerprint:'{}'})""")
    alts = fulltext_elements(store, gkey, "cart", "link", "el:b")
    assert [a["gid"] for a in alts] == ["el:a"]
    w = MapWriter(store, gkey, Settings())
    w.bump_locator("el:b", "css=#b2")
    row = store.ro_query(gkey, "MATCH (x:Element {gid:'el:b'}) RETURN x.locators, x.locator_hits")[0]
    assert row[0][0] == "css=#b2" and json.loads(row[1]) == {"css=#b2": 1}


def test_fulltext_alternatives_are_never_riskier(store, gkey):
    """Healing 'Cart' through the graph must not pick 'Empty cart': a broken safe element is never replaced by a
    riskier one (that would bypass the approval the riskier one needs)."""
    ensure_device_schema(store, gkey, 384)
    store.query(gkey, """CREATE (:Element {gid:'el:cart', role:'button', name_sample:'Cart', risk_cost:1, locators:['css=#c'], fingerprint:'{}'}),
                                (:Element {gid:'el:empty', role:'button', name_sample:'Empty cart', risk_cost:10000, locators:['css=#e'], fingerprint:'{}'}),
                                (:Element {gid:'el:view', role:'button', name_sample:'View cart', risk_cost:0, locators:['css=#v'], fingerprint:'{}'})""")
    alts = fulltext_elements(store, gkey, "Cart", "button", "el:cart", max_risk_cost=1)
    assert [a["gid"] for a in alts] == ["el:view"]


@pytest.mark.browser
def test_graph_and_vision_healing(site_url):
    with BrowserSession() as s:
        s.goto(site_url + "/index.html")
        alt = {"gid": "el:alt", "role": "button", "locators": ["role=button|name=Account"]}
        r = ElementResolver(s, graph=lambda el: [alt])
        found = r.resolve({"gid": "el:x", "role": "button", "name_sample": "Acct", "locators": ["css=#nope"],
                           "fingerprint": {}})
        assert found and found.method == "graph:0"
        llm = FakeLLM(lambda sys, u: json.dumps({"index": 0}))
        r2 = ElementResolver(s, llm=llm)
        found2 = r2.resolve({"gid": "el:y", "role": "button", "name_sample": "zzz", "locators": ["css=#nope"],
                             "fingerprint": {"role": "button", "name": "zzz", "gcss": "nowhere", "bbox": [900, 900, 5, 5]}},
                            reference_png=s.screenshot_png())
        assert found2 and found2.method == "vision" and llm.images[-1] is not None


@pytest.mark.browser
def test_locators_only_still_skips_graph_and_fuzzy_matches(site_url):
    """Learned optional steps (agent-layer A5) resolve by name-bearing locators only, never by graph alternatives."""
    with BrowserSession() as s:
        s.goto(site_url + "/index.html")
        alt = {"gid": "el:alt", "role": "button", "locators": ["role=button|name=Account"]}
        r = ElementResolver(s, graph=lambda el: [alt])
        assert r.resolve({"gid": "el:x", "role": "button", "name_sample": "Acct", "locators": ["css=#account-btn"],
                          "fingerprint": {}}, locators_only=True) is None          # css is not name-bearing
        hit = r.resolve({"gid": "el:x", "role": "button", "name_sample": "Account",
                         "locators": ["role=button|name=Account"], "fingerprint": {}}, locators_only=True)
        assert hit and hit.method == "locator:0"
