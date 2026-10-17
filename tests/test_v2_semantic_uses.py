import pytest
from sitegraph.agents.navigator import Navigator
from sitegraph.browser.session import BrowserSession
from sitegraph.capture.model import ElementObs
from sitegraph.embed import HashEmbedder
from sitegraph.llm import FakeLLM
from sitegraph.ops import Ops
from sitegraph.semantic.distiller import Distiller
from sitegraph.semantic.uses import exploration_priority, parity_report, pattern_of_component, port_workflow
from sitegraph.tools import ToolContext, build_toolbox
from tests.test_v2_semantic import responder, two_devices  # noqa: F401

pytestmark = [pytest.mark.integration, pytest.mark.browser]


def test_port_workflow_to_phone_and_run(store, msettings, two_devices, site_url):
    prof, keys, domain = two_devices
    Distiller(store, msettings.uid, HashEmbedder(), FakeLLM(responder), msettings, prof).rebuild()
    wf = store.ro_query(keys["desktop"], "MATCH (w:Workflow) RETURN w.gid LIMIT 1")[0][0]
    ported = port_workflow(store, msettings.uid, HashEmbedder(), domain, wf, "desktop", "phone")
    assert store.ro_query(keys["phone"], "MATCH (w:Workflow {gid:$w}) RETURN w.source", {"w": ported}) == [["ported"]]
    with BrowserSession("phone") as s:
        res = Navigator(store, keys["phone"], domain, s, msettings, HashEmbedder()).execute_workflow(ported)
        assert res.status == "done" and s.url.endswith("/cart.html")
    assert isinstance(parity_report(store, msettings.uid), list)
    comp = store.ro_query(keys["desktop"], "MATCH (k:Component {kind:'navigation'}) RETURN k.gid LIMIT 1")[0][0]
    assert pattern_of_component(store, msettings.uid, keys["desktop"], comp)
    prio = exploration_priority(store, msettings.uid, HashEmbedder())
    el = ElementObs(gid="el:x", cs_gid="cs:x", role="link", name_norm="cart (#)", name_sample="Cart (0)", tag="a",
                    locators=[], fingerprint={}, risk="navigate", risk_cost=0)
    assert isinstance(prio(el), float)


def test_priority_orders_the_shared_work_queue(store, settings):
    ops = Ops(store, settings.uid, "prio.test")
    ops.register_agent("agent-p", "navigator")
    a = ops.add_item("scope-p", "g", "click", "http://prio.test/", 0, el_gid="el:a", priority=0.1)
    b = ops.add_item("scope-p", "g", "click", "http://prio.test/", 0, el_gid="el:b", priority=0.9)
    first, _ = ops.next_item("scope-p", "agent-p", "g")
    assert first["gid"] == b                                  # higher priority first, same depth


def test_semantic_tools(store, msettings, two_devices):
    prof, keys, domain = two_devices
    ctx = ToolContext(settings=msettings, store=store, key=keys["desktop"], domain=domain, embedder=HashEmbedder(),
                      llm=FakeLLM(responder), profile=prof)
    box = build_toolbox(ctx)
    assert {"semantic_rebuild", "semantic_parity", "port_workflow"} <= set(box.names())
    assert box.call("semantic_rebuild", {}).ok
    assert box.call("semantic_parity", {}).ok
    bad = box.call("port_workflow", {"workflow": "wf:none", "from_device": "desktop", "to_device": "phone"})
    assert not bad.ok and "no actions" in bad.error
    ctx.close()
