import asyncio
import pytest
from mcp import Client
from sitegraph.embed import HashEmbedder
from sitegraph.mcp_server import build_server
from sitegraph.ops import Ops
from sitegraph.schema import ensure_device_schema
from sitegraph.tools import ToolContext, build_toolbox

pytestmark = pytest.mark.integration


@pytest.fixture
def box(store, gkey, settings):
    ensure_device_schema(store, gkey, 384)
    store.query(gkey, """CREATE (p:Page {gid:'pg:1', url_template:'/', example_urls:['http://s.test/']}),
        (a:State:Entry {gid:'st:a', summary:'home', untried_count:2}), (b:State:Frontier {gid:'st:b', summary:'cart', untried_count:1}),
        (p)-[:HAS_STATE]->(a),
        (a)-[:ACTION {gid:'ac:1', label:'click Cart', el_gid:'el:1', reliability:0.9, seen_count:5, risk_cost:0,
                      pre_cstates:[], requires_inputs:[]}]->(b),
        (k:Component {gid:'cp:1'})-[:HAS_CSTATE]->(cs:ComponentState {gid:'cs:1'})-[:CONTAINS]->(e:Element {gid:'el:1', name_sample:'Cart'}),
        (ws:WStep {gid:'ws:1'})-[:ON]->(e), (ws)-[:STEP_OF]->(w:Workflow {gid:'wf:1', name:'buy'}),
        (:Storyboard {gid:'sb:1', title:'How to buy'})-[:BASED_ON]->(w)""")
    ctx = ToolContext(settings=settings, store=store, key=gkey, domain="s.test", embedder=HashEmbedder())
    yield build_toolbox(ctx)
    ctx.close()


def test_graph_intelligence_tools_work_without_a_transport(box):
    assert {"next_actions", "frontier", "impact", "git_log", "blame", "submit_task", "task_status"} <= set(box.names())
    nxt = box.call("next_actions", {})
    assert nxt.ok and nxt.data[0]["label"] == "click Cart"
    assert nxt.to_message_text().startswith("[untrusted website content")        # labels come from pages
    assert [f["gid"] for f in box.call("frontier", {}).data] == ["st:b"]
    hit = box.call("impact", {"component_gid": "cp:1"})
    assert hit.ok and hit.data == [{"element": "Cart", "workflows": ["buy"], "demos": ["How to buy"]}]
    assert not box.call("submit_task", {"text": "explore"}).ok          # no orchestrator in this context


def test_submit_and_follow_a_task(box, store, settings):
    class FakeOrchestrator:
        def submit(self, text, domain, device="desktop", navigators=None):
            return Ops(store, settings.uid, domain).create_task("chat", text, device=device)

    box.ctx.extra["orchestrator"] = FakeOrchestrator()
    task = box.call("submit_task", {"text": "explore the site"})
    assert task.ok
    status = box.call("task_status", {"task_id": task.data["task"]})
    assert status.ok and status.data["status"] == "queued" and status.data["request"] == "explore the site"


def test_the_new_tools_are_served_over_mcp(box):
    async def go():
        async with Client(build_server(box)) as client:
            return {t.name for t in (await client.list_tools()).tools}

    assert {"next_actions", "frontier", "impact", "git_log", "submit_task"} <= asyncio.run(go())
