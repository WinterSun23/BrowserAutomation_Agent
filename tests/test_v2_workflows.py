import time
import pytest
from sitegraph.agents.curator import Curator
from sitegraph.agents.orchestrator import Orchestrator
from sitegraph.agents.workflows import compose_workflow, load_workflow_steps, save_workflow_from_actions
from sitegraph.blobs import BlobStore
from sitegraph.config import Settings
from sitegraph.embed import HashEmbedder
from sitegraph.events import NullEvents
from sitegraph.graph.activity import ActivityLog
from sitegraph.graph.queries import find_targets
from sitegraph.llm import FakeLLM
from sitegraph.schema import ensure_device_schema

pytestmark = pytest.mark.integration


@pytest.fixture
def g(store, gkey):
    ensure_device_schema(store, gkey, 384)
    store.query(gkey, """CREATE (a:State {gid:'st:a'}), (b:State {gid:'st:b'}), (c:State {gid:'st:c'}),
        (:Element {gid:'el:1'}), (:Element {gid:'el:2'}),
        (a)-[:ACTION {gid:'ac:1', el_gid:'el:1', label:'go b', reliability:0.9, cost:1.2, risk_cost:0, pre_cstates:[], requires_inputs:[]}]->(b),
        (b)-[:ACTION {gid:'ac:2', el_gid:'el:2', label:'go c', reliability:0.9, cost:1.2, risk_cost:0, pre_cstates:[], requires_inputs:[]}]->(c)""")
    return gkey


def test_save_from_actions_and_compose(store, g):
    e = HashEmbedder()
    w1 = save_workflow_from_actions(store, g, e, ["ac:1"], "to b", "go to b")
    w2 = save_workflow_from_actions(store, g, e, ["ac:2"], "to c", "go to c from b")
    wc = compose_workflow(store, g, e, [w1, w2], "a to c", "go all the way to c")
    steps = load_workflow_steps(store, g, wc)
    assert [s["action_gid"] for s in steps] == ["ac:1", "ac:2"]
    assert store.ro_query(g, "MATCH (:Workflow {gid:$w})<-[:STEP_OF]-(:WStep)-[:CALLS]->(x:Workflow) RETURN count(x)",
                          {"w": wc}) == [[2]]


def test_followed_by_similar_rank_and_intent_dedupe(store, g):
    e = HashEmbedder()
    w1 = save_workflow_from_actions(store, g, e, ["ac:1"], "open page b", "open page b")
    w2 = save_workflow_from_actions(store, g, e, ["ac:2"], "open page b now", "open page b now")
    log = ActivityLog(store, g)
    for wf in (w1, w2, w1, w2):
        time.sleep(0.003)  # distinct start times keep FOLLOWED_BY ordering deterministic
        r = log.start_run("execute", "n")
        log.link_workflow(r, wf)
        log.finish_run(r, "done")
    cur = Curator(store, g, Settings(workflow_similar=0.7, intent_dedupe=0.9), e, FakeLLM())
    cur.link_workflows()
    cur.similar_workflows()
    cur.workflow_rank()
    assert store.ro_query(g, "MATCH (:Workflow {gid:$a})-[f:FOLLOWED_BY]->(:Workflow {gid:$b}) RETURN f.count",
                          {"a": w1, "b": w2}) == [[2]]
    assert store.ro_query(g, "MATCH (:Workflow)-[s:SIMILAR_TO]->(:Workflow) RETURN count(s) > 0") == [[True]]
    assert store.ro_query(g, "MATCH (w:Workflow) WHERE w.pagerank IS NULL RETURN count(w)") == [[0]]
    store.query(g, "CREATE (:Intent {gid:'in:x', text:'open page b', hits:5, emb: vecf32($v)})",
                {"v": e.embed(["open page b"])[0]})
    merged = cur.dedupe_intents()
    assert merged >= 1
    assert store.ro_query(g, "MATCH (i:Intent) WHERE i.text = 'open page b' RETURN count(i)") == [[1]]


def test_find_targets_reports_the_intent_and_recommend_lists_next_workflows(store, g, settings):
    e = HashEmbedder()
    store.query(g, "MATCH (s:State {gid:'st:b'}) SET s.summary = 'page b', s.emb = vecf32($v)", {"v": e.embed(["page b"])[0]})
    store.query(g, "CREATE (i:Intent {gid:'in:b', text:'open page b', hits:0, emb: vecf32($v)}) "
                   "WITH i MATCH (s:State {gid:'st:b'}) CREATE (i)-[:FULFILLED_BY {score:1.0}]->(s)",
                {"v": e.embed(["open page b"])[0]})
    best = find_targets(store, g, e.embed(["open page b"])[0], k=3)[0]
    assert best.gid == "st:b" and best.intent == "in:b"
    save_workflow_from_actions(store, g, e, ["ac:1"], "to b", "go to b")
    o = Orchestrator(settings, store, e, FakeLLM(), BlobStore(settings.blob_dir), NullEvents())

    class Site:                       # recommend() only needs the graph key
        key, domain, device = g, "x.test", "desktop"

    assert o.recommend(Site())["next_workflows"][0]["name"] == "to b"
