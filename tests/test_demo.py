import json
import pytest
from sitegraph.agents.curator import Curator
from sitegraph.agents.navigator import Navigator
from sitegraph.blobs import BlobStore
from sitegraph.browser.session import BrowserSession
from sitegraph.db import device_key
from sitegraph.demo.director import DemoDirector
from sitegraph.demo.render import render_clickthrough_html, render_guide_html
from sitegraph.embed import HashEmbedder
from sitegraph.llm import FakeLLM
from sitegraph.normalize import domain_of
from sitegraph.policy import ExplorationPolicy
from sitegraph.schema import ensure_device_schema
from tests.helpers import site_responder


def test_html_renders_are_self_contained():
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 10
    slides = [("Click Products", png, [10, 10, 50, 20]), ("Done", png, None)]
    assert "data:image/png;base64" in render_guide_html("Demo", slides)
    html = render_clickthrough_html("Demo", slides, (1280, 720))
    assert "hotspot" in html and "Click Products" in html


def _responder(interpretation):
    def r(system, user):
        if user.startswith("Request:"):
            return json.dumps(interpretation)
        return site_responder(system, user)
    return r


@pytest.fixture(scope="module")
def world(store, msettings, site_url, tmp_path_factory):
    settings = msettings
    domain = domain_of(site_url)
    key = device_key(settings.uid, domain, "desktop")
    ensure_device_schema(store, key, 384)
    with BrowserSession() as s:
        Navigator(store, key, domain, s, settings, HashEmbedder()).explore(site_url + "/index.html",
                                                                           ExplorationPolicy(max_depth=2))
    Curator(store, key, settings, HashEmbedder(), FakeLLM(site_responder)).run_all(1)
    return store, settings, key, domain, BlobStore(tmp_path_factory.mktemp("blobs"))


@pytest.mark.integration
@pytest.mark.browser
@pytest.mark.slow
@pytest.mark.parametrize("interp", [
    {"formats": ["gif", "screenshot"], "scope_kind": "transition", "target": "account menu"},
    {"formats": ["video", "guide", "clickthrough"], "scope_kind": "goal", "target": "open cart"},
])
def test_director_produces_renders(world, interp):
    store, settings, key, domain, blobs = world
    d = DemoDirector(store, key, domain, settings, HashEmbedder(), FakeLLM(_responder(interp)), blobs)
    res = d.run("make a demo")
    assert {f for f, _ in res.renders} == set(interp["formats"])
    for _, ref in res.renders:
        assert blobs.path(ref).stat().st_size > 0
    rows = store.ro_query(key, "MATCH (:Storyboard {gid:$g})-[:RENDERED_AS]->(r:Render) RETURN r.status",
                          {"g": res.storyboard_gid})
    assert rows and all(r[0] == "ok" for r in rows)
    assert store.ro_query(key, "MATCH (:Storyboard {gid:$g})<-[:IN_STORYBOARD]-(:Shot)-[:FROM_STEP]->(:Step) "
                               "RETURN count(*) > 0", {"g": res.storyboard_gid}) == [[True]]
