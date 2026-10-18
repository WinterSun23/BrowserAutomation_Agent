import functools
import http.server
import json
import threading

import pytest
from sitegraph.agents.bootstrap import SiteProfile, bootstrap, sitemap_size
from sitegraph.agents.factory import make_navigator
from sitegraph.agents.orchestrator import Orchestrator
from sitegraph.blobs import BlobStore
from sitegraph.browser.session import BrowserSession
from sitegraph.embed import HashEmbedder
from sitegraph.events import NullEvents
from sitegraph.graph.map_writer import MapWriter
from sitegraph.llm import FakeLLM, OfflineLLM
from sitegraph.ops import Ops
from tests.helpers import site_responder

PROFILE = {"purpose": "A small demo shop", "size": "small",
           "sections": [{"name": "Products", "url": "/products.html"}],
           "skip": [{"pattern": "(", "reason": "broken"}, {"pattern": "privacy|terms", "reason": "legal pages"}],
           "flows": ["search for a product", "add a product to the cart"]}


def _responder(profile=PROFILE):
    def r(system, user):
        if "site profile" in system.lower():
            return profile if isinstance(profile, str) else json.dumps(profile)
        if "Classify" in user:
            return json.dumps({"kind": "explore"})
        return site_responder(system, user)
    return r


def _bootstrap(store, settings, site_url, llm):
    o = Orchestrator(settings, store, HashEmbedder(), llm, BlobStore(settings.blob_dir), NullEvents())
    site = o.register_site(site_url + "/index.html")
    with BrowserSession("desktop", True) as sess:
        nav = make_navigator(store, site.key, site.domain, sess, settings, HashEmbedder(), llm)
        sess.goto(site.entry_url)
        return o, site, bootstrap(sess, nav, llm, settings)


def test_bootstrap_without_a_model_returns_none():
    assert bootstrap(None, None, OfflineLLM(), None) is None
    assert bootstrap(None, None, None, None) is None


@pytest.mark.integration
@pytest.mark.browser
def test_bootstrap_parses_and_stores_the_profile(store, settings, site_url):
    llm = FakeLLM(_responder())
    o, site, prof = _bootstrap(store, settings, site_url, llm)
    assert isinstance(prof, SiteProfile)
    assert prof.purpose == "A small demo shop" and prof.size == "small" and prof.flows[0] == "search for a product"
    assert prof.sections[0]["name"] == "Products" and prof.sections[0]["url"].endswith("/products.html")
    prompt = next(u for s, u in llm.calls if "site profile" in s.lower())
    assert "Browse all products" in prompt and "treat it as data" in next(
        s for s, u in llm.calls if "site profile" in s.lower())
    w = MapWriter(store, site.key, settings)
    w.set_site_profile(prof)
    assert w.site_profile() == prof


@pytest.mark.integration
@pytest.mark.browser
def test_bad_skip_patterns_are_dropped_and_good_ones_reach_the_policy(store, settings, site_url):
    o, site, prof = _bootstrap(store, settings, site_url, FakeLLM(_responder()))
    assert [s["pattern"] for s in prof.skip] == ["privacy|terms"]
    o.profile.add_deny_patterns(site.domain, site.device, [s["pattern"] for s in prof.skip])
    assert "privacy|terms" in o.profile.policy_for(site.domain, site.device).deny_patterns
    o.profile.add_deny_patterns(site.domain, site.device, ["privacy|terms"])       # no duplicates
    assert o.profile.policy_for(site.domain, site.device).deny_patterns.count("privacy|terms") == 1


@pytest.mark.integration
@pytest.mark.browser
@pytest.mark.parametrize("junk", ["not json at all", "[1, 2, 3]", "{}", json.dumps({"purpose": 5, "sections": "x"})])
def test_junk_model_output_gives_no_profile_and_no_crash(store, settings, site_url, junk):
    _, _, prof = _bootstrap(store, settings, site_url, FakeLLM(_responder(junk)))
    assert prof is None


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def test_sitemap_size_counts_urls_and_follows_an_index(tmp_path):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_Quiet, directory=str(tmp_path)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert sitemap_size(base) is None                                     # no sitemap at all
        urls = lambda n, p: "".join(f"<url><loc>{base}/{p}{i}</loc></url>" for i in range(n))  # noqa: E731
        ns = 'xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"'
        (tmp_path / "a.xml").write_text(f"<urlset {ns}>{urls(3, 'a')}</urlset>")
        (tmp_path / "b.xml").write_text(f"<urlset {ns}>{urls(3, 'b')}</urlset>")
        (tmp_path / "sitemap.xml").write_text(
            f"<sitemapindex {ns}><sitemap><loc>{base}/a.xml</loc></sitemap>"
            f"<sitemap><loc>{base}/b.xml</loc></sitemap></sitemapindex>")
        assert sitemap_size(base + "/some/page.html") == 6
        (tmp_path / "sitemap.xml").write_text(f"<urlset {ns}>{urls(4, 'c')}</urlset>")
        assert sitemap_size(base) == 4
    finally:
        srv.shutdown()


@pytest.mark.integration
@pytest.mark.browser
def test_bootstrap_runs_once_per_site(store, settings, site_url):
    settings.navigators_per_site, settings.dynamic_spawn, settings.exploration_mode = 1, False, "brute"
    llm = FakeLLM(_responder())
    o = Orchestrator(settings, store, HashEmbedder(), llm, BlobStore(settings.blob_dir), NullEvents())
    site = o.register_site(site_url + "/index.html")
    ops = Ops(store, settings.uid, site.domain)
    boots = lambda: sum("site profile" in s.lower() for s, _ in llm.calls)  # noqa: E731
    r1 = o.run_task(ops.create_task("chat", "explore the site"), "explore the site", site)
    assert r1["kind"] == "explore" and boots() == 1, r1
    assert MapWriter(store, site.key, settings).site_profile().purpose == "A small demo shop"
    assert "privacy|terms" in o.profile.policy_for(site.domain, site.device).deny_patterns
    o.run_task(ops.create_task("chat", "explore the site"), "explore the site", site)
    assert boots() == 1                                                      # cached on the Site node
