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
    expected = set(interp["formats"]) | ({"guide_md"} if "guide" in interp["formats"] else set())   # a guide also comes as Markdown
    assert {f for f, _ in res.renders} == expected
    for _, ref in res.renders:
        assert blobs.path(ref).stat().st_size > 0
    rows = store.ro_query(key, "MATCH (:Storyboard {gid:$g})-[:RENDERED_AS]->(r:Render) RETURN r.status",
                          {"g": res.storyboard_gid})
    assert rows and all(r[0] == "ok" for r in rows)
    assert store.ro_query(key, "MATCH (:Storyboard {gid:$g})<-[:IN_STORYBOARD]-(:Shot)-[:FROM_STEP]->(:Step) "
                               "RETURN count(*) > 0", {"g": res.storyboard_gid}) == [[True]]


@pytest.mark.integration
@pytest.mark.browser
@pytest.mark.slow
def test_make_demo_over_mcp_after_a_browser_tool(world):
    """Every MCP call runs on one worker thread; the browser tools have already started Playwright there, so the
    demo recorder must not start a second one (it used to fail with 'Playwright Sync API inside the asyncio loop')."""
    import asyncio

    from mcp import Client

    from sitegraph.mcp_server import build_server
    from sitegraph.tools import ToolContext, build_toolbox

    store, settings, key, domain, blobs = world
    interp = {"formats": ["screenshot"], "scope_kind": "goal", "target": "open cart"}
    ctx = ToolContext(settings=settings, store=store, key=key, domain=domain, embedder=HashEmbedder(),
                      llm=FakeLLM(_responder(interp)), blobs=blobs)
    box = build_toolbox(ctx)
    site = store.ro_query(key, "MATCH (p:Page) RETURN p.example_urls[0] LIMIT 1")[0][0]

    server = build_server(box)

    async def go():
        async with Client(server) as client:
            opened = await client.call_tool("open_url", {"url": site})
            demo = await client.call_tool("make_demo", {"request": "make a demo"})
            return opened, demo

    try:
        opened, demo = asyncio.run(go())
    finally:
        server.close_tools()
    assert not opened.is_error
    text = "\n".join(getattr(c, "text", "") for c in demo.content)
    assert not demo.is_error, text
    assert '"format": "screenshot"' in text
