"""Tools that wrap the deterministic engine (route execution, exploration, demos) and the user's memory."""
from __future__ import annotations

import re

from ..agents.curator import Curator
from ..agents.escalation import make_curator_escalation
from ..agents.orchestrator import MAX_NAVIGATORS, Orchestrator, SiteInfo
from ..agents.workflows import save_workflow_from_run
from ..db import ops_key
from ..agents.demo_agent import DemoAgent
from ..events import NullEvents
from ..graph.queries import entry_state
from .browser_tools import get_nav
from .core import ToolContext, ToolResult, ToolSpec, Toolbox
from .graph_tools import policy_of

_SENSITIVE_MEMORY = re.compile(
    r"(password|passwd|passcode|api[ _-]?key|secret|token|credit card|card number|ssn|\b[\w.+-]+@[\w-]+\.[\w.]+\b|\d{9,})",
    re.I)


def _run_goal(ctx: ToolContext, args: dict):
    nav = get_nav(ctx)
    res = nav.execute_goal(args["goal"], {}, policy_of(ctx), args.get("item"), request_text=args["goal"])
    ctx.extra["last_run"] = res.run_gid
    ctx.extra.pop("elements", None)
    return {"status": res.status, "explanation": res.explanation, "url": nav.session.url, "heals": res.heals,
            "replans": res.replans}


def _run_route(ctx: ToolContext, args: dict):
    route = ctx.extra.get("routes", {}).get(args["route_id"])
    if route is None:
        raise ValueError(f"unknown route_id '{args['route_id']}': call plan_route first")
    nav = get_nav(ctx)
    run = nav.activity.start_run("execute", "chat-agent", request=f"route {args['route_id']}")
    counter, outcomes, status = [0], [], "done"
    if route.steps:
        status, _, _, _, _ = nav.drive_to(route.steps[0].src, run, counter, {}, policy_of(ctx))
    if status == "done":
        for step in route.steps:
            out = nav.execute_step(step, run, counter[0], {}, None)
            counter[0] += 1
            outcomes.append({"step": step.label, "ok": out.ok, "healed": out.healed})
            if not out.ok:
                status = "failed"
                break
    nav.activity.finish_run(run, status)
    ctx.extra["last_run"] = run
    ctx.extra.pop("elements", None)
    return {"status": status, "steps": outcomes, "url": nav.session.url}


def _explore(ctx: ToolContext, args: dict):
    entry = args.get("entry_url") or (entry_state(ctx.store, ctx.key) or [None, None])[1]
    if not entry:
        raise ValueError("no entry_url given and the site has not been explored before")
    # resources come from config: the agent can ask for fewer browsers but never for more than allowed
    cap = max(1, min(MAX_NAVIGATORS, ctx.settings.navigators_per_site))
    n = max(1, min(int(args.get("navigators") or cap), cap))
    orch = Orchestrator(ctx.settings, ctx.store, ctx.embedder, ctx.llm, ctx.blobs, NullEvents())
    site = SiteInfo(ctx.domain, ctx.device, ctx.key, ops_key(ctx.settings.uid, ctx.domain), entry)
    policy = policy_of(ctx)
    rep, agents = orch._explore(site, policy, "chat-explore", n)
    stats = Curator(ctx.store, ctx.key, ctx.settings, ctx.embedder, ctx.llm).run_all(
        policy.max_risk_cost, escalate=make_curator_escalation(ctx.settings, ctx.store, ctx.key, ctx.embedder, ctx.llm))
    return {"states": len(rep.states), "actions": rep.actions, "transitions": rep.transitions,
            "blocked_as_risky": rep.blocked, "navigators": n, "agents": agents, **stats}


def _make_demo(ctx: ToolContext, args: dict):
    """Hand the request to the Demo agent (or, without a tool-calling model, the fixed pipeline). The agent may ask a
    clarifying question: the chat then ends its turn with it, and the answer comes back here as `reply`."""
    agent = ctx.extra.get("demo_agent")
    if args.get("reply") and agent is not None:
        res = agent.reply(args["reply"])
    else:
        if not args.get("request"):
            raise ValueError("give the request for the demo (or a reply to the demo agent's question)")
        agent = DemoAgent(ctx.settings, ctx.store, ctx.embedder, ctx.llm, ctx.blobs, ctx.key, ctx.domain, ctx.device,
                          ctx.profile, policy_of(ctx))
        ctx.extra["demo_agent"] = agent
        res = agent.run(args["request"])
    if res.status == "needs_input":
        return ToolResult(True, data={"question": res.answer}, pause=res.answer)
    ctx.extra.pop("demo_agent", None)
    out = {"status": res.status, "storyboard": res.storyboard_gid,
           "renders": [{"format": f, "file": ref} for f, ref in res.renders], "summary": res.answer}
    return out if res.status == "done" else ToolResult(False, data=out, error=res.answer)


def _save_workflow(ctx: ToolContext, args: dict):
    run = ctx.extra.get("last_run")
    if not run:
        raise ValueError("nothing to save yet: run_goal or run_route first")
    wf = save_workflow_from_run(ctx.store, ctx.key, ctx.embedder, run, args["name"], args["goal"])
    return {"workflow": wf}


def _profile(ctx: ToolContext):
    if ctx.profile is None:
        raise ValueError("memory is not available in this session")
    return ctx.profile


def _remember(ctx: ToolContext, args: dict):
    text = args["text"].strip()
    if _SENSITIVE_MEMORY.search(text):
        raise ValueError("refused: memories must not contain passwords, keys, emails, card or long numbers or other "
                         "personal data. Save only preferences or notes about a site.")
    gid = _profile(ctx).remember(text, kind=args.get("kind", "semantic"), importance=0.5,
                                 ref_graph=ctx.key if args.get("about_this_site") else None)
    return {"saved": text, "memory": gid, "note": "Tell the user what you saved; they can ask to forget it."}


def _recall(ctx: ToolContext, args: dict):
    return [{"text": t, "similarity": round(s, 3)} for t, s in _profile(ctx).recall(args["query"], int(args.get("k", 5)))]


def _list_memories(ctx: ToolContext, args: dict):
    return _profile(ctx).list_memories()


def _forget(ctx: ToolContext, args: dict):
    return {"forgotten": _profile(ctx).forget(args["memory"])}


def _s(**props) -> dict:
    return {"type": "object", "properties": props}


def register(box: Toolbox) -> None:
    box.register(ToolSpec(
        "run_goal", "Use the saved site map to take the browser to a screen described in words, e.g. 'the Mystery "
        "category' or 'shopping cart'. Plans a safe route, runs it and repairs broken steps itself. Pass `item` to pick "
        "an entry of a repeated list. Cheaper and more reliable than clicking by hand when the screen is known.",
        {"type": "object", "properties": {"goal": {"type": "string"}, "item": {"type": "string"}},
         "required": ["goal"]}, _run_goal, "act", untrusted_output=True))
    box.register(ToolSpec(
        "run_route", "Run a route returned by plan_route.", {"type": "object", "properties": {
            "route_id": {"type": "string"}}, "required": ["route_id"]}, _run_route, "act", untrusted_output=True))
    box.register(ToolSpec(
        "explore", "Map the site with several browsers at once (only safe actions; respects robots.txt). Use when the "
        "site is unknown or a screen cannot be found. The number of browsers is limited by the user's configuration.",
        {"type": "object", "properties": {"entry_url": {"type": "string"}, "navigators": {"type": "integer",
                                                                                         "minimum": 1}}},
        _explore, "act"))
    box.register(ToolSpec(
        "make_demo", "Record and render a demo of something on the site (video, gif, screenshot, step guide, "
        "click-through) from a plain-language request. If it returns a question, ask the user and call make_demo again "
        "with their answer as `reply`.", {"type": "object", "properties": {
            "request": {"type": "string"}, "reply": {"type": "string"}}}, _make_demo, "act"))
    box.register(ToolSpec(
        "save_workflow", "Save the last successful run_goal/run_route as a reusable named workflow.",
        {"type": "object", "properties": {"name": {"type": "string"}, "goal": {"type": "string"}},
         "required": ["name", "goal"]}, _save_workflow, "act"))
    box.register(ToolSpec(
        "remember", "Save a durable preference or note about the user or a site (e.g. 'prefers short GIFs', "
        "'example.com shows a cookie wall'), only when the user asks or it is clearly useful. Never save secrets, "
        "typed values or personal data. Tell the user what you saved.",
        {"type": "object", "properties": {"text": {"type": "string"}, "about_this_site": {"type": "boolean"}},
         "required": ["text"]}, _remember, "act"))
    box.register(ToolSpec("recall", "Search saved memories by meaning.", {"type": "object", "properties": {
        "query": {"type": "string"}, "k": {"type": "integer", "minimum": 1, "maximum": 10}}, "required": ["query"]},
                          _recall, "read"))
    box.register(ToolSpec("list_memories", "List everything saved in memory (so the user can review it).", _s(),
                          _list_memories, "read"))
    box.register(ToolSpec("forget", "Delete one memory by its gid (from list_memories).", {
        "type": "object", "properties": {"memory": {"type": "string"}}, "required": ["memory"]}, _forget, "act"))
