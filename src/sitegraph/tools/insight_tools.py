"""Read-only graph-intelligence tools: what to do next, what is unexplored, what a change would affect. They are part
of the shared tool layer, so the chat agent and every MCP client get them. (`git_log` and `blame` live in git_tools.)"""
from __future__ import annotations

from ..graph.queries import entry_state, frontier_states, impact_of_component, next_actions
from .core import ToolContext, ToolSpec, Toolbox
from .graph_tools import policy_of


def next_actions_tool(ctx: ToolContext, args: dict):
    state = args.get("state_gid") or (entry_state(ctx.store, ctx.key) or [None])[0]
    return next_actions(ctx.store, ctx.key, state, policy_of(ctx).max_risk_cost, int(args.get("k", 5))) if state else []


def frontier(ctx: ToolContext, args: dict):
    return frontier_states(ctx.store, ctx.key, limit=int(args.get("limit", 10)))


def impact(ctx: ToolContext, args: dict):
    return impact_of_component(ctx.store, ctx.key, args["component_gid"])


def register(box: Toolbox) -> None:
    box.register(ToolSpec(
        "next_actions", "The most useful actions from the entry screen (or `state_gid`) ranked by how often they were "
        "seen and how reliable they are, within the allowed risk level.",
        {"type": "object", "properties": {"state_gid": {"type": "string"},
                                          "k": {"type": "integer", "minimum": 1, "maximum": 20}}},
        next_actions_tool, "read", untrusted_output=True))
    box.register(ToolSpec(
        "frontier", "Screens that still have untried elements, reachable from the entry screen: where exploring "
        "next would find the most.", {"type": "object", "properties": {"limit": {"type": "integer", "minimum": 1,
                                                                                  "maximum": 50}}},
        frontier, "read", untrusted_output=True))
    box.register(ToolSpec(
        "impact", "What depends on a component (gid): its elements with the saved workflows and demos that use them, "
        "so the effect of a site change can be judged.",
        {"type": "object", "properties": {"component_gid": {"type": "string"}}, "required": ["component_gid"]},
        impact, "read", untrusted_output=True))
