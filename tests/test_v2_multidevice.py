import json
import pytest
from sitegraph.agents.orchestrator import Orchestrator
from sitegraph.blobs import BlobStore
from sitegraph.embed import HashEmbedder
from sitegraph.events import NullEvents
from sitegraph.llm import FakeLLM
from sitegraph.ops import Ops
from sitegraph.tools import ToolContext, build_toolbox
from tests.helpers import site_responder

pytestmark = [pytest.mark.integration, pytest.mark.browser]


def _orch(store, settings):
    llm = FakeLLM(lambda s, u: json.dumps({"kind": "explore"}) if "Classify" in u else site_responder(s, u))
    return Orchestrator(settings, store, HashEmbedder(), llm, BlobStore(settings.blob_dir), NullEvents())


def test_explore_fans_out_to_every_device(store, settings, site_url):
    o = _orch(store, settings)
    sites = o.register_devices(site_url + "/index.html", ["desktop", "phone"])
    ops = Ops(store, settings.uid, sites[0].domain)
    parent = ops.create_task("multi", "explore the site")
    result = o.run_multi(parent, "explore the site", sites)
    assert set(result["devices"]) == {"desktop", "phone"}
    for s in sites:
        assert store.ro_query(s.key, "MATCH (st:State) RETURN count(st) > 0") == [[True]]
    assert ops.get_task(parent)["status"] == "done"
    assert ops.q("MATCH (:Task)-[:SUBTASK_OF]->(:Task {id:$p}) RETURN count(*)", p=parent) == [[2]]
    # the browsers are split between the devices, not multiplied by them (navigators_per_site=2 over 2 devices)
    assert [result["devices"][d]["navigators"] for d in ("desktop", "phone")] == [1, 1]


def test_the_chat_tools_can_explore_every_device(store, settings, site_url):
    from urllib.parse import urlsplit

    ctx = ToolContext(settings=settings, store=store, key="unused", domain=urlsplit(site_url).netloc,
                      embedder=HashEmbedder(), llm=_orch(store, settings).llm)
    box = build_toolbox(ctx)
    assert "run_on_devices" in box.names()
    res = box.call("run_on_devices", {"text": "explore the site", "devices": ["desktop", "tablet"],
                                      "entry_url": site_url + "/index.html"})
    assert res.ok, res.error
    assert set(res.data["devices"]) == {"desktop", "tablet"} and res.data["task"]
    ctx.close()
