import json
import pytest
from sitegraph.agents.orchestrator import _keyword_kind
from sitegraph.db import device_key
from sitegraph.embed import HashEmbedder
from sitegraph.llm import FakeLLM
from sitegraph.normalize import domain_of
from sitegraph.policy import ExplorationPolicy
from sitegraph.profile import Profile
from sitegraph.schema import ensure_device_schema
from sitegraph.tools import ToolContext, build_toolbox

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def memory_vault(monkeypatch):
    monkeypatch.setenv("SITEGRAPH_VAULT", "memory")


def test_secret_never_written_to_graph(store, settings):
    p = Profile(store, settings.uid, HashEmbedder())
    p.set_secret("email", "alice@example.com", scope="shop.test")
    assert p.secret_for("email", "shop.test") == "alice@example.com"
    assert p.secret_for("email", "other.test") is None
    for k in store.list_graphs(f"u:{settings.uid}:"):
        rows = store.ro_query(k, "MATCH (n) RETURN properties(n)")
        assert "alice@example.com" not in json.dumps(rows)
    assert p.secret_vars() == [{"var": "email", "scope": "shop.test"}]


def test_site_policy_inferred_prefs_and_session_summary(store, settings):
    p = Profile(store, settings.uid, HashEmbedder())
    p.register_site("shop.test", "desktop", "k", "o", "http://shop.test/")
    p.save_policy(ExplorationPolicy(name="shop-forms", allow_forms=True), domain="shop.test", device="desktop")
    assert p.policy_for("shop.test", "desktop").allow_forms is True
    assert p.policy_for("other.test", "desktop").allow_forms is False
    sid = p.start_session()
    for _ in range(3):
        p.log_request(sid, "gif of menu", "done", kind="demo", meta={"formats": ["gif"]})
    p.infer_demo_format()
    assert p.get_prefs()["demo.format"] == "gif"
    p.set_pref("demo.format", "video")
    p.infer_demo_format()
    assert p.get_prefs()["demo.format"] == "video"
    p.end_session(sid, FakeLLM(lambda s, u: json.dumps({"summary": "made three gifs"})))
    assert store.ro_query(p.key, "MATCH (s:Session {id:$i}) RETURN s.summary", {"i": sid}) == [["made three gifs"]]


def test_remember_is_routed_first():
    assert _keyword_kind("remember that I prefer gifs") == "remember"
    assert _keyword_kind("make a gif") == "demo"


def test_secret_and_policy_tools_are_gated_and_values_stay_hidden(store, settings):
    p = Profile(store, settings.uid, HashEmbedder())
    p.register_site("shop.test", "desktop", "k", "o", "http://shop.test/")
    ctx = ToolContext(settings=settings, store=store, key="k", domain="shop.test", embedder=HashEmbedder(),
                      llm=FakeLLM(), profile=p)
    box = build_toolbox(ctx)
    args = {"var": "email", "value": "bob@example.com", "scope": "shop.test"}
    first = box.call("set_secret", args)
    assert not first.ok and first.needs_approval and "bob@example.com" not in json.dumps(first.needs_approval) and p.secret_for("email", "shop.test") is None
    assert box.redacted("set_secret", args)["value"] == "<hidden>"
    ctx.approvals.approve(first.needs_approval["id"])
    assert box.call("set_secret", args).ok and p.secret_for("email", "shop.test") == "bob@example.com"
    listed = box.call("list_secrets", {})
    assert listed.ok and "bob@example.com" not in listed.to_message_text() and "email" in listed.to_message_text()
    pol = {"policy": {"name": "forms", "allow_forms": True}, "this_site_only": True}
    denied = box.call("set_policy", pol)
    assert denied.needs_approval and not p.policy_for("shop.test", "desktop").allow_forms
    ctx.approvals.approve(denied.needs_approval["id"])
    assert box.call("set_policy", pol).ok and p.policy_for("shop.test", "desktop").allow_forms
    ctx.close()


@pytest.mark.browser
def test_fill_from_secret_types_value_without_exposing_it(store, settings, site_url):
    domain = domain_of(site_url)
    key = device_key(settings.uid, domain, "desktop")
    ensure_device_schema(store, key, 384)
    p = Profile(store, settings.uid, HashEmbedder())
    p.set_secret("query", "zebra-secret-99", scope=domain)
    ctx = ToolContext(settings=settings, store=store, key=key, domain=domain, embedder=HashEmbedder(),
                      llm=FakeLLM(), profile=p)
    box = build_toolbox(ctx)
    try:
        assert box.call("open_url", {"url": site_url + "/index.html"}).ok
        obs = box.call("observe", {}).data
        sid = next(e["id"] for e in obs["elements"] if e["role"] == "searchbox")
        r = box.call("fill", {"element_id": sid, "from_secret": "query"})
        assert r.ok, r.error
        assert "zebra-secret-99" not in r.to_message_text() and r.data["variable"]
        page = ctx.browser().page
        assert page.input_value("input[type=search], [role=searchbox], input[name=q]") == "zebra-secret-99"
        assert not box.call("fill", {"element_id": sid, "from_secret": "missing"}).ok
        assert not box.call("fill", {"element_id": sid}).ok          # neither value nor from_secret
        assert "zebra-secret-99" in ctx.extra["typed"]               # kept in memory so it is scrubbed from observations
    finally:
        ctx.close()
