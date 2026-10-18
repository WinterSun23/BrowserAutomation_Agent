"""Read-only tools over the site graph: find screens, plan routes, describe, recommend."""
from __future__ import annotations

from ..graph.queries import (best_routes, entry_state, explain, find_targets, next_actions, recommendations,
                             shared_type_summary)
from ..policy import ExplorationPolicy
from .core import ToolContext, ToolSpec, Toolbox


def policy_of(ctx: ToolContext) -> ExplorationPolicy:
    return ctx.policy or ExplorationPolicy()


def _obj(**props) -> dict:
    return {"type": "object", "properties": props}


def find_screens(ctx: ToolContext, args: dict):
    vec = ctx.embedder.embed([args["query"]])[0]
    hits = find_targets(ctx.store, ctx.key, vec, int(args.get("k", 5)), text=args["query"])
    out = []
    for t in hits:
        hit = {"kind": t.kind, "gid": t.gid, "description": t.text, "score": round(t.score, 3)}
        if t.item:
            # one state stands for every entry of a list and carries the name of whichever entry was seen last
            hit["item"] = t.item
            hit["description"] = (shared_type_summary(t.text, t.item) +
                                  f" To open it: run_goal with goal '{args['query']}' and item '{t.item}'.")
        out.append(hit)
    return out


def plan_route(ctx: ToolContext, args: dict):
    src = args.get("from_state") or (entry_state(ctx.store, ctx.key) or [None])[0]
    if not src:
        raise ValueError("the site has not been explored yet; run explore first")
    routes = best_routes(ctx.store, ctx.key, src, args["to_state"], policy_of(ctx).max_risk_cost)
    if not routes:
        return {"route": None, "reason": "no route within the allowed risk level (risky steps are excluded)"}
    stored = ctx.extra.setdefault("routes", {})
    rid = f"r{len(stored) + 1}"
    stored[rid] = routes[0]
    return {"route_id": rid, "explanation": explain(routes[0]), "steps": len(routes[0].steps),
            "success_probability": round(routes[0].probability, 3)}


def list_known_sites(ctx: ToolContext, args: dict):
    return ctx.profile.sites() if ctx.profile else []


def describe_state(ctx: ToolContext, args: dict):
    rows = ctx.store.ro_query(ctx.key, """
        MATCH (p:Page)-[:HAS_STATE]->(s:State {gid:$g})
        RETURN p.url_template, p.title, p.example_titles, p.example_urls, coalesce(s.summary, ''),
               coalesce(s.untried_count, 0)""", {"g": args["state_gid"]})
    if not rows:
        raise ValueError(f"unknown state {args['state_gid']}")
    tpl, title, titles, urls, summary, untried = rows[0]
    return {"url_pattern": tpl, "title": title, "example_titles": titles, "example_urls": urls, "summary": summary,
            "untried_elements": untried,
            "next_actions": next_actions(ctx.store, ctx.key, args["state_gid"], policy_of(ctx).max_risk_cost, 8)}


def recommend(ctx: ToolContext, args: dict):
    return recommendations(ctx.store, ctx.key, policy_of(ctx).max_risk_cost)


def list_workflows(ctx: ToolContext, args: dict):
    rows = ctx.store.ro_query(ctx.key, "MATCH (w:Workflow) RETURN w.gid, w.name, w.goal, coalesce(w.run_count, 0), "
                                       "coalesce(w.success_rate, 0.0) ORDER BY w.run_count DESC")
    return [dict(gid=r[0], name=r[1], goal=r[2], runs=r[3], success_rate=r[4]) for r in rows]


def register(box: Toolbox) -> None:
    box.register(ToolSpec(
        "find_screens", "Search the known site map for screens (or saved workflows) matching a description or a name "
        "(for example 'shopping cart', or an item name such as 'Mystery'). Returns candidates with gids and scores. A "
        "candidate with `item` is a shared page type (one screen for every entry of a list): open it with run_goal "
        "and that `item`.",
        {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer", "minimum": 1,
                                                                             "maximum": 10}},
         "required": ["query"]}, find_screens, "read", untrusted_output=True))
    box.register(ToolSpec(
        "plan_route", "Plan the cheapest route over known, allowed (non-risky) actions from the entry screen (or "
        "from_state) to a screen gid from find_screens. Returns a route_id for run_route and a readable explanation.",
        {"type": "object", "properties": {"to_state": {"type": "string"}, "from_state": {"type": "string"}},
         "required": ["to_state"]}, plan_route, "read", untrusted_output=True))
    box.register(ToolSpec("list_known_sites", "List the sites the user has explored.", _obj(), list_known_sites, "read"))
    box.register(ToolSpec(
        "describe_state", "Describe a known screen: page type, example titles, summary, untried elements and likely "
        "next actions.", {"type": "object", "properties": {"state_gid": {"type": "string"}},
                          "required": ["state_gid"]}, describe_state, "read", untrusted_output=True))
    box.register(ToolSpec("recommend", "Suggest next actions, unexplored areas, workflows worth saving and demo "
                          "candidates for this site.", _obj(), recommend, "read", untrusted_output=True))
    box.register(ToolSpec("list_workflows", "List saved workflows (reusable automations) for this site.", _obj(),
                          list_workflows, "read"))
