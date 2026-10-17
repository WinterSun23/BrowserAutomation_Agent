import json
from dataclasses import replace

import pytest
from sitegraph.agents.curator import Curator
from sitegraph.agents.escalation import CuratorEscalation, make_curator_escalation
from sitegraph.config import Settings
from sitegraph.ids import gid
from sitegraph.embed import HashEmbedder
from sitegraph.llm import ChatMessage, FakeLLM, OfflineLLM, ScriptedLLM, ToolCall
from sitegraph.schema import ensure_device_schema
from sitegraph.tools.core import Toolbox, ToolContext
from sitegraph.tools import curator_tools

pytestmark = pytest.mark.integration


@pytest.fixture
def seeded(store, gkey, settings):
    """One page with two near-identical states (a, b) and a third (d); a failing action; a second page (c)."""
    ensure_device_schema(store, gkey, 384)
    store.query(gkey, """
      CREATE (p:Page {gid:'pg:1', url_template:'/product', title:'Product', example_titles:['Product']}),
             (q:Page {gid:'pg:2', url_template:'/cart', title:'Cart', example_titles:['Cart']}),
             (a:State:Entry {gid:'st:a', summary:'Product page', tokens:['t1','t2','t3','t4','t5','t6','t7'],
                             blocking_sig:'', seen_count:3, first_seen:100, last_seen:300, aliases:[]}),
             (b:State {gid:'st:b', summary:'Product page', tokens:['t1','t2','t3','t4','t5','t6','t8','t9'],
                       blocking_sig:'', seen_count:2, first_seen:50, last_seen:400, aliases:[]}),
             (d:State {gid:'st:d', summary:'Dialog', tokens:['x1'], blocking_sig:'dlg', seen_count:1,
                       first_seen:1, last_seen:1, aliases:[]}),
             (c:State {gid:'st:c', summary:'Cart', tokens:['c1','c2'], blocking_sig:'', seen_count:4,
                       first_seen:1, last_seen:1, aliases:[]}),
             (p)-[:HAS_STATE]->(a), (p)-[:HAS_STATE]->(b), (p)-[:HAS_STATE]->(d), (q)-[:HAS_STATE]->(c),
             (a)-[:ACTION {gid:$old, el_gid:'el:cart', action_type:'click', label:'click cart', seen_count:4,
                           success_count:3, first_seen:100, last_seen:300, reliability:0.67, cost:1.8,
                           risk:'navigate', risk_cost:0, pre_cstates:[], requires_inputs:[]}]->(c),
             (b)-[:ACTION {gid:'ac:dup', el_gid:'el:cart', action_type:'click', label:'click cart', seen_count:2,
                           success_count:2, first_seen:50, last_seen:400, reliability:0.75, cost:1.6,
                           risk:'navigate', risk_cost:0, pre_cstates:[], requires_inputs:[]}]->(c),
             (a)-[:ACTION {gid:'ac:bad', el_gid:'el:flaky', action_type:'click', label:'click flaky', seen_count:9,
                           success_count:1, first_seen:1, last_seen:1, reliability:0.18, cost:4.4,
                           risk:'navigate', risk_cost:0, pre_cstates:[], requires_inputs:[]}]->(c),
             (:Intent {gid:'in:1', text:'open product'})-[:FULFILLED_BY {score:1.0}]->(b)""",
                {"old": gid("ac", "st:a", "el:cart", "click", "", "st:c")})      # the real edge-gid formula
    return store, gkey, settings


def _curator(seeded, **overrides):
    store, key, settings = seeded
    return Curator(store, key, replace(settings, **overrides), HashEmbedder(), OfflineLLM())


def _tools(seeded, curator, **overrides):
    store, key, settings = seeded
    ctx = ToolContext(settings=replace(settings, **overrides), store=store, key=key, embedder=HashEmbedder())
    ctx.extra.update(findings=curator.find_anomalies(), curator=curator, changes=0)
    box = Toolbox(ctx)
    curator_tools.register(box)
    return box


def call(name, args=None, id="c"):
    return ChatMessage("assistant", tool_calls=[ToolCall(id, name, args or {})])


def test_find_anomalies_reports_each_kind(seeded):
    cur = _curator(seeded, anomaly_states_per_page=2)
    kinds = {f["kind"] for f in cur.find_anomalies()}
    assert {"near_duplicate", "clashing_names", "state_explosion", "failing_edge"} <= kinds
    dup = next(f for f in cur.find_anomalies() if f["kind"] == "near_duplicate")
    assert set(dup["states"]) == {"st:a", "st:b"} and 0.6 <= dup["similarity"] < 0.85    # 6 shared of 9 tokens
    bad = next(f for f in cur.find_anomalies() if f["kind"] == "failing_edge")
    assert bad["action"] == "ac:bad" and bad["reliability"] < 0.4
    assert "st:d" not in str([f for f in cur.find_anomalies() if f["kind"] == "near_duplicate"])   # modal state ignored


def test_a_clean_graph_has_no_findings(store, gkey, settings):
    ensure_device_schema(store, gkey, 384)
    store.query(gkey, "CREATE (:Page {gid:'pg:1', url_template:'/a'})-[:HAS_STATE]->(:State {gid:'st:1', "
                      "summary:'A fairly descriptive summary', tokens:['a','b'], blocking_sig:''})")
    assert Curator(store, gkey, settings, HashEmbedder(), OfflineLLM()).find_anomalies() == []


def test_merge_states_moves_edges_sums_counters_and_logs(seeded):
    store, key, _ = seeded
    box = _tools(seeded, _curator(seeded))
    res = box.call("merge_states", {"keep": "st:a", "merge": "st:b", "reason": "same product page, one extra link"})
    assert res.ok, res.error
    assert store.ro_query(key, "MATCH (s:State {gid:'st:b'}) RETURN count(s)") == [[0]]            # folded in
    keep = store.ro_query(key, "MATCH (s:State {gid:'st:a'}) RETURN s.aliases, s.seen_count, s.tokens")[0]
    assert "st:b" in keep[0] and keep[1] == 5 and "t9" in keep[2]                                   # alias, history, tokens
    edges = store.ro_query(key, "MATCH (:State {gid:'st:a'})-[r:ACTION]->(:State {gid:'st:c'}) "
                                "WHERE r.el_gid = 'el:cart' RETURN count(r), max(r.seen_count), max(r.success_count)")
    assert edges == [[1, 6, 5]]                                                                    # two edges became one
    assert store.ro_query(key, "MATCH (:Intent {gid:'in:1'})-[:FULFILLED_BY]->(s:State) RETURN s.gid") == [["st:a"]]
    log = store.ro_query(key, "MATCH (o:Observation {kind:'curation'}) RETURN o.detail")
    detail = json.loads(log[0][0])
    assert detail["op"] == "merge_states" and detail["merged"] == "st:b" and "same product page" in detail["reason"]
    assert detail["merged_state"]["summary"] == "Product page"                                      # undo information kept


def test_merge_refuses_different_pages_and_modal_states(seeded):
    box = _tools(seeded, _curator(seeded))
    assert "different pages" in box.call("merge_states", {"keep": "st:a", "merge": "st:c", "reason": "x"}).error
    assert "modal dialog" in box.call("merge_states", {"keep": "st:a", "merge": "st:d", "reason": "x"}).error
    assert "different states" in box.call("merge_states", {"keep": "st:a", "merge": "st:a", "reason": "x"}).error


def test_quarantine_and_restore_round_trip_changes_what_the_planner_sees(seeded):
    store, key, _ = seeded
    box = _tools(seeded, _curator(seeded))
    assert box.call("quarantine_edge", {"action_gid": "ac:bad", "reason": "fails 8 of 9 times"}).ok
    assert store.ro_query(key, "MATCH ()-[r:ACTION {gid:'ac:bad'}]->() RETURN count(r)") == [[0]]
    broken = store.ro_query(key, "MATCH ()-[r:BROKEN_ACTION {gid:'ac:bad'}]->() RETURN r.reason, r.seen_count")
    assert broken == [["fails 8 of 9 times", 9]]
    assert not box.call("quarantine_edge", {"action_gid": "ac:bad", "reason": "again"}).ok          # already quarantined
    assert box.call("restore_edge", {"action_gid": "ac:bad", "reason": "was a one-off"}).ok
    assert store.ro_query(key, "MATCH ()-[r:ACTION {gid:'ac:bad'}]->() RETURN r.seen_count") == [[9]]
    assert store.ro_query(key, "MATCH ()-[r:BROKEN_ACTION]->() RETURN count(r)") == [[0]]


def test_rename_state_updates_summary_and_logs(seeded):
    store, key, _ = seeded
    box = _tools(seeded, _curator(seeded))
    assert box.call("rename_state", {"state_gid": "st:c", "summary": "Shopping cart listing the chosen items",
                                     "reason": "title only said Cart"}).ok
    assert store.ro_query(key, "MATCH (s:State {gid:'st:c'}) RETURN s.summary") == [["Shopping cart listing the chosen items"]]
    assert json.loads(store.ro_query(key, "MATCH (o:Observation {kind:'curation'}) RETURN o.detail")[0][0])["old_summary"] == "Cart"


def test_the_number_of_edits_per_review_is_limited(seeded):
    box = _tools(seeded, _curator(seeded), curator_max_changes=1)
    assert box.call("rename_state", {"state_gid": "st:c", "summary": "first", "reason": "r"}).ok
    second = box.call("rename_state", {"state_gid": "st:a", "summary": "second", "reason": "r"})
    assert not second.ok and "change limit" in second.error


def test_the_review_agent_merges_through_tools_and_the_curator_recomputes(seeded):
    store, key, settings = seeded
    llm = ScriptedLLM([call("list_findings", id="c1"),
                       call("compare_states", {"state_a": "st:a", "state_b": "st:b"}, "c2"),
                       call("merge_states", {"keep": "st:a", "merge": "st:b", "reason": "same page, one extra link"}, "c3"),
                       ChatMessage("assistant", content="Merged st:b into st:a; left the failing action alone.")])
    cur = Curator(store, key, settings, HashEmbedder(), OfflineLLM())
    stats = cur.run_all(1, escalate=CuratorEscalation(llm, settings, store, key, HashEmbedder()))
    assert stats["curation"] == "done" and stats["curation_changes"] == 1 and stats["anomalies"] >= 2
    assert "Merged st:b" in stats["curation_summary"]
    assert store.ro_query(key, "MATCH (s:State {gid:'st:b'}) RETURN count(s)") == [[0]]
    assert store.ro_query(key, "MATCH (s:State) WHERE s.pagerank IS NULL RETURN count(s)") == [[0]]    # recomputed after


def test_without_escalation_run_all_changes_nothing(seeded):
    store, key, settings = seeded
    stats = Curator(store, key, settings, HashEmbedder(), OfflineLLM()).run_all(1)
    assert "curation" not in stats
    assert store.ro_query(key, "MATCH (s:State) RETURN count(s)") == [[4]]


def test_make_curator_escalation_needs_a_tool_calling_llm():
    assert make_curator_escalation(Settings(), None, "k", HashEmbedder(), OfflineLLM()) is None
    assert make_curator_escalation(Settings(), None, "k", HashEmbedder(), FakeLLM(lambda s, u: "")) is None
    assert make_curator_escalation(Settings(escalate_curator=False), None, "k", HashEmbedder(), ScriptedLLM([])) is None
    assert isinstance(make_curator_escalation(Settings(), None, "k", HashEmbedder(), ScriptedLLM([])), CuratorEscalation)


def test_curator_tool_edits_are_recorded_by_the_context_writer(seeded):
    from sitegraph.graph.recording import RecordingMapWriter

    store, key, settings = seeded
    cur = _curator(seeded)
    box = _tools(seeded, cur)
    box.ctx.writer = RecordingMapWriter(store, key, settings)
    assert box.call("merge_states", {"keep": "st:a", "merge": "st:b", "reason": "same page"}).ok
    assert box.call("quarantine_edge", {"action_gid": "ac:bad", "reason": "keeps failing"}).ok
    assert box.call("rename_state", {"state_gid": "st:a", "summary": "A product screen", "reason": "clearer"}).ok
    assert box.call("restore_edge", {"action_gid": "ac:bad", "reason": "was fine"}).ok
    assert [c.op for c in box.ctx.writer.changes] == ["merge_states", "quarantine", "rename_state", "unquarantine"]
    assert store.ro_query(key, "MATCH (s:State {gid:'st:b'}) RETURN count(s)") == [[0]]
    assert store.ro_query(key, "MATCH (s:State {gid:'st:a'}) RETURN s.summary") == [["A product screen"]]
