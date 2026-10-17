"""Tools of the Curator escalation agent: inspect suspicious parts of the map and make a few reasoned, logged edits
(merge near-duplicate states, rename/re-summarise, quarantine or restore failing actions).

Every edit needs a `reason`, is counted against `Settings.curator_max_changes`, and is recorded as an
`Observation{kind:'curation'}` that keeps enough detail (including the removed node's properties) to undo it by hand.
"""
from __future__ import annotations

import json
from itertools import combinations

from ..capture.segment import jaccard
from ..graph.activity import ActivityLog
from ..graph.map_writer import MapWriter
from ..ids import now_ms
from .core import ToolContext, ToolSpec, Toolbox

_DROP_PROPS = {"emb"}                                    # not worth keeping in a log


def _props(entity) -> dict:
    return {k: v for k, v in dict(entity.properties).items() if k not in _DROP_PROPS}


def _budget(ctx: ToolContext) -> None:
    ctx.extra["changes"] = ctx.extra.get("changes", 0) + 1
    if ctx.extra["changes"] > ctx.settings.curator_max_changes:
        raise ValueError(f"change limit reached ({ctx.settings.curator_max_changes} edits per review); stop and "
                         "summarise what is left for a human")


def _log(ctx: ToolContext, op: str, target_gid: str, reason: str, **detail) -> None:
    ActivityLog(ctx.store, ctx.key).add_observation(
        "curation", {"op": op, "reason": reason, "ts": now_ms(), **detail}, of_gid=target_gid, of_label="State")


def list_findings(ctx: ToolContext, args: dict):
    return ctx.extra.get("findings", [])


def inspect_state(ctx: ToolContext, args: dict):
    rows = ctx.store.ro_query(ctx.key, """
        MATCH (p:Page)-[:HAS_STATE]->(s:State {gid:$g})
        OPTIONAL MATCH (s)-[:COMPOSED_OF]->(:ComponentState)-[:CONTAINS]->(e:Element)
        RETURN p.url_template, p.title, coalesce(p.example_titles, []), s.summary, s.seen_count, s.aliases,
               size(coalesce(s.tokens, [])), collect(DISTINCT e.name_sample)""", {"g": args["state_gid"]})
    if not rows:
        raise ValueError(f"unknown state {args['state_gid']}")
    tpl, title, titles, summary, seen, aliases, ntokens, names = rows[0]
    edges = ctx.store.ro_query(ctx.key, """
        MATCH (s:State {gid:$g}) OPTIONAL MATCH (s)-[o:ACTION]->() OPTIONAL MATCH ()-[i:ACTION]->(s)
        RETURN count(DISTINCT o), count(DISTINCT i)""", {"g": args["state_gid"]})[0]
    return {"page": tpl, "page_title": title, "example_titles": titles, "summary": summary, "times_seen": seen,
            "aliases": aliases, "token_count": ntokens, "element_names": [n for n in names if n][:25],
            "outgoing_actions": edges[0], "incoming_actions": edges[1]}


def compare_states(ctx: ToolContext, args: dict):
    rows = ctx.store.ro_query(ctx.key, "MATCH (s:State) WHERE s.gid IN $g RETURN s.gid, coalesce(s.tokens, []), s.summary",
                              {"g": [args["state_a"], args["state_b"]]})
    found = {r[0]: r for r in rows}
    if len(found) != 2:
        raise ValueError("unknown state gid")
    a, b = set(found[args["state_a"]][1]), set(found[args["state_b"]][1])
    return {"similarity": round(jaccard(a, b), 3), "only_in_a": sorted(a - b)[:20], "only_in_b": sorted(b - a)[:20],
            "summary_a": found[args["state_a"]][2], "summary_b": found[args["state_b"]][2]}


def state_edges(ctx: ToolContext, args: dict):
    rows = ctx.store.ro_query(ctx.key, """
        MATCH (s:State {gid:$g})-[r:ACTION|BROKEN_ACTION]->(t:State)
        RETURN type(r), r.gid, r.label, r.seen_count, r.success_count, r.reliability, t.gid
        ORDER BY r.reliability LIMIT 40""", {"g": args["state_gid"]})
    return [dict(kind=r[0], gid=r[1], label=r[2], seen=r[3], successes=r[4], reliability=r[5], to=r[6]) for r in rows]


def _writer(ctx: ToolContext):
    """The context's (possibly recording) writer, so every edit is a replayable, revertible operation."""
    return ctx.writer or MapWriter(ctx.store, ctx.key, ctx.settings)


def merge_states(ctx: ToolContext, args: dict):
    keep, merge, reason = args["keep"], args["merge"], args["reason"]
    if keep == merge:
        raise ValueError("keep and merge must be different states")
    _budget(ctx)
    out = _writer(ctx).merge_states(keep, merge)
    _log(ctx, "merge_states", keep, reason, merged=merge, edges_moved=out["actions_moved"],
         merged_state=json.loads(json.dumps(out["merged_state"], default=str)))
    return {"kept": keep, "merged": merge, "actions_moved": out["actions_moved"], "aliases": out["aliases"]}


def rename_state(ctx: ToolContext, args: dict):
    _budget(ctx)
    rows = ctx.store.ro_query(ctx.key, """MATCH (p:Page)-[:HAS_STATE]->(s:State {gid:$g})
                                          RETURN coalesce(s.summary, ''), coalesce(p.example_titles, [])""",
                              {"g": args["state_gid"]})
    if not rows:
        raise ValueError(f"unknown state {args['state_gid']}")
    old, titles = rows[0]
    summary = args["summary"][:300]
    titles, state_vec, intents = ctx.extra["curator"].state_embeddings(summary, [], titles)
    _writer(ctx).rename_state(args["state_gid"], summary, titles, state_vec, intents)
    _log(ctx, "rename_state", args["state_gid"], args["reason"], old_summary=old, new_summary=summary)
    return {"state": args["state_gid"], "summary": summary}


def quarantine_edge(ctx: ToolContext, args: dict):
    _budget(ctx)
    rows = ctx.store.ro_query(ctx.key, "MATCH (a:State)-[r:ACTION {gid:$g}]->(b:State) RETURN a.gid, b.gid",
                              {"g": args["action_gid"]})
    if not rows:
        raise ValueError(f"unknown action {args['action_gid']} (it may already be quarantined)")
    src, dst = rows[0]
    _writer(ctx).quarantine("ACTION", args["action_gid"], args["reason"][:300])
    _log(ctx, "quarantine_edge", src, args["reason"], action=args["action_gid"], to=dst)
    return {"quarantined": args["action_gid"], "note": "route planning now ignores it; restore_edge undoes this"}


def restore_edge(ctx: ToolContext, args: dict):
    _budget(ctx)
    rows = ctx.store.ro_query(ctx.key, "MATCH (a:State)-[r:BROKEN_ACTION {gid:$g}]->(b:State) RETURN a.gid, b.gid",
                              {"g": args["action_gid"]})
    if not rows:
        raise ValueError(f"no quarantined action {args['action_gid']}")
    src, dst = rows[0]
    _writer(ctx).unquarantine("ACTION", args["action_gid"])
    _log(ctx, "restore_edge", src, args["reason"], action=args["action_gid"], to=dst)
    return {"restored": args["action_gid"]}


def _obj(required=(), **props) -> dict:
    return {"type": "object", "properties": props, "required": list(required)}


_S = {"type": "string"}


def register(box: Toolbox) -> None:
    box.register(ToolSpec("list_findings", "The anomalies the curator program found (near-duplicate states, clashing "
                          "or generic names, state explosion, failing actions).", _obj(), list_findings, "read"))
    box.register(ToolSpec("inspect_state", "Describe one state: page, example titles, summary, element names, how often "
                          "it was seen, actions in and out.", _obj(["state_gid"], state_gid=_S), inspect_state, "read",
                          untrusted_output=True))
    box.register(ToolSpec("compare_states", "Compare the structure of two states: similarity and what only one has.",
                          _obj(["state_a", "state_b"], state_a=_S, state_b=_S), compare_states, "read",
                          untrusted_output=True))
    box.register(ToolSpec("state_edges", "List the actions leaving a state with their reliability.",
                          _obj(["state_gid"], state_gid=_S), state_edges, "read", untrusted_output=True))
    box.register(ToolSpec(
        "merge_states", "Merge two states of the SAME page that are truly the same screen: `merge` is folded into `keep` "
        "(actions, history and references move over; the removed gid stays as an alias). Only when sure; give the reason.",
        _obj(["keep", "merge", "reason"], keep=_S, merge=_S, reason=_S), merge_states, "act"))
    box.register(ToolSpec(
        "rename_state", "Replace a state's summary with a clearer one (it is re-embedded for matching).",
        _obj(["state_gid", "summary", "reason"], state_gid=_S, summary=_S, reason=_S), rename_state, "act"))
    box.register(ToolSpec(
        "quarantine_edge", "Take a failing action out of route planning (kept for history). Use for actions that keep "
        "failing, not for ones that merely look odd.", _obj(["action_gid", "reason"], action_gid=_S, reason=_S),
        quarantine_edge, "act"))
    box.register(ToolSpec("restore_edge", "Put a quarantined action back into route planning.",
                          _obj(["action_gid", "reason"], action_gid=_S, reason=_S), restore_edge, "act"))
