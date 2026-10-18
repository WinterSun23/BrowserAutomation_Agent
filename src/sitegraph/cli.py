from __future__ import annotations

import argparse
import logging
import builtins
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .agents.execution_agent import ExecutionAgent
from .agents.loop import AgentRun
from .agents.orchestrator import Orchestrator
from .policy import ExplorationPolicy
from .blobs import BlobStore
from .config import Settings, load_settings
from .db import GraphStore
from .embed import BgeEmbedder, CachedEmbedder, HashEmbedder
from .events import ConsoleEvents, NullEvents
from .llm import LiteLLMClient, OfflineLLM, TracingLLM, can_call_tools, has_model
from .ops import Ops
from .tools import ToolContext, build_toolbox


def make_llm(settings: Settings):
    """A real model when LLM_API_KEY (or LLM_API_BASE for a local server) is set, otherwise the offline fallback."""
    if settings.llm_api_key or settings.llm_api_base:
        return LiteLLMClient(settings.llm_model, settings.llm_api_base, settings.llm_api_key)
    return OfflineLLM()


@dataclass
class Runtime:
    settings: Settings
    store: GraphStore
    embedder: object
    llm: object
    blobs: BlobStore


def build_runtime(args: argparse.Namespace, quiet: bool = False) -> Runtime | None:
    """Everything the commands share: settings, the graph store, and the embedder and LLM wrapped for caching and
    tracing. Returns None (after saying why) when FalkorDB cannot be reached."""
    settings = load_settings(args.config)
    if args.uid:
        settings.uid = args.uid
    try:
        store = GraphStore(settings.falkor_host, settings.falkor_port)
        store.redis.ping()
    except Exception as exc:
        print(f"Cannot reach FalkorDB at {settings.falkor_host}:{settings.falkor_port} ({exc}).\n"
              "Start it with: docker compose up -d", file=sys.stderr)
        return None
    llm = make_llm(settings)
    if not has_model(llm) and not quiet:
        print("No LLM configured: using built-in defaults (keyword routing, plain summaries and captions). "
              "Set LLM_API_KEY to enable a model.", file=sys.stderr)
    if args.embedder == "bge":
        if not quiet:
            print(f"Loading embedding model {settings.embed_model} (downloads on first use)...", file=sys.stderr)
        embedder = BgeEmbedder(settings.embed_model)
    else:
        embedder = HashEmbedder(settings.embed_dim)
    # sensitive tool arguments (typed values, secrets) are hidden in the per-run traces
    redact = build_toolbox(ToolContext(settings=settings)).redacted
    return Runtime(settings, store, CachedEmbedder(embedder, settings.data_dir / "embed_cache.sqlite"),
                   TracingLLM(llm, settings.data_dir / "traces", redact=redact), BlobStore(settings.blob_dir))


def _policy_flags(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("exploration policy (override config.yaml `policy:` and the saved policy for this run)")
    g.add_argument("--max-depth", type=int, help="how many page hops from the entry page to follow")
    g.add_argument("--max-states", type=int, help="stop after this many screens")
    g.add_argument("--max-actions", type=int, help="stop after this many recorded actions and transitions")
    g.add_argument("--budget", type=int, help="time budget for one exploration, in seconds")


def apply_policy_flags(profile, args: argparse.Namespace) -> None:
    """Command-line policy flags beat the config.yaml block, which beats the saved policies. Raises ValueError on a
    bad value. Nothing is written to the graph."""
    flags = {"max_depth": args.max_depth, "max_states": args.max_states, "max_actions": args.max_actions,
             "budget_s": args.budget}
    profile.policy_overrides = {**profile.policy_overrides, **{k: v for k, v in flags.items() if v is not None}}
    ExplorationPolicy().with_overrides(profile.policy_overrides)


def setup_logging(verbosity: int) -> None:
    """0: warnings only, 1 (-v): info, 2 (-vv): debug."""
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s", force=True)
    logging.getLogger().setLevel(level)


def _verbose_flag(p: argparse.ArgumentParser) -> None:
    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="-v: progress lines (states, steps, explorers) on stderr and info logs; -vv: debug logs")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m sitegraph", description="SiteGraph: explore, automate and demo a website.")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="register a site and run one or more requests against it, in order")
    r.add_argument("site", help="entry URL of a public website, e.g. https://books.toscrape.com/")
    r.add_argument("requests", nargs="+", help='e.g. "explore the site" "open cart" "make a gif of the menu"')
    r.add_argument("--device", default="desktop", choices=["desktop", "tablet", "phone"])
    r.add_argument("--navigators", type=int, help="how many agents explore in parallel (default: config.yaml)")
    r.add_argument("--kind", choices=["explore", "automate", "demo", "recommend", "save_workflow", "remember", "ask"],
                   help="treat every request as this kind instead of letting the model or keywords decide")
    r.add_argument("--rescan", action="store_true",
                   help="when exploring, visit elements that earlier runs already tried")
    r.add_argument("--mode", choices=["auto", "brute", "guided"],
                   help="exploration mode for this run (default: config.yaml exploration_mode)")
    r.add_argument("--embedder", default="bge", choices=["bge", "hash"],
                   help="bge = real semantic model (downloads ~130 MB once); hash = offline, weak, for quick tests")
    r.add_argument("--config", default="config.yaml")
    r.add_argument("--uid", help="user id that namespaces the graphs (default: config.yaml)")
    _policy_flags(r)
    _verbose_flag(r)
    c = sub.add_parser("chat", help="talk to an agent that explores, operates and demos the site with tools")
    c.add_argument("site", help="entry URL of a public website")
    c.add_argument("--device", default="desktop", choices=["desktop", "tablet", "phone"])
    c.add_argument("--embedder", default="bge", choices=["bge", "hash"])
    c.add_argument("--config", default="config.yaml")
    c.add_argument("--uid", help="user id that namespaces the graphs (default: config.yaml)")
    _policy_flags(c)
    _verbose_flag(c)
    c.add_argument("--no-agent", action="store_true", help="skip the LLM agent and use the fixed request pipeline")
    m = sub.add_parser("mcp", help="serve the tools over MCP (stdio) so an MCP client such as Claude Desktop can use them")
    m.add_argument("site", help="entry URL of a public website")
    m.add_argument("--device", default="desktop", choices=["desktop", "tablet", "phone"])
    m.add_argument("--embedder", default="bge", choices=["bge", "hash"])
    m.add_argument("--config", default="config.yaml")
    m.add_argument("--uid", help="user id that namespaces the graphs (default: config.yaml)")
    _verbose_flag(m)
    e = sub.add_parser("eval", help="run an evaluation suite (eval/suites/*.yaml) and write a report")
    e.add_argument("suite", help="suite file, e.g. eval/suites/offline.yaml")
    e.add_argument("--variant", action="append", default=[], metavar="NAME:key=value,...",
                   help="settings overrides for one variant, e.g. brute:exploration_mode=brute (repeatable; "
                        "default: the suite's `variants:`, else one variant with the configured settings)")
    e.add_argument("--repeat", type=int, default=1, help="runs of every task per variant (flaky tasks show up)")
    e.add_argument("--out", help="output directory (default: <data_dir>/eval/<run-id>)")
    e.add_argument("--keep", action="store_true", help="keep each task run's graphs (default: deleted)")
    e.add_argument("--site-url", help="fills {site} in the suite (default: the suite's `serve:` directory)")
    e.add_argument("--compare", metavar="DIR", help="an earlier run directory: adds deltas and regressions")
    e.add_argument("--embedder", default="bge", choices=["bge", "hash"])
    e.add_argument("--config", default="config.yaml")
    _verbose_flag(e)
    sub.add_parser("serve", help="web UI (not built yet)")
    return p


def run(args: argparse.Namespace) -> int:
    rt = build_runtime(args)
    if rt is None:
        return 2
    settings, store, embedder, llm = rt.settings, rt.store, rt.embedder, rt.llm
    if getattr(args, "mode", None):
        settings.exploration_mode = args.mode
    try:
        orch = Orchestrator(settings, store, embedder, llm, rt.blobs, _events(args))
        apply_policy_flags(orch.profile, args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    site = orch.register_site(args.site, args.device)
    ops = Ops(store, settings.uid, site.domain)
    failed = False
    orch.on_intent = lambda i: print(f"-> {i['kind']} (goal: {i['goal']}; chosen by {i['source']})", file=sys.stderr)
    for text in args.requests:
        print(f"\n== {text}", file=sys.stderr)
        task = ops.create_task("chat", text, device=args.device)
        result = orch.run_task(task, text, site, args.navigators, kind=args.kind, rescan=args.rescan)
        print(json.dumps(result, indent=2, default=str))
        failed = failed or result.get("kind") == "error"
    print(f"\nGraphs: {site.key}, {site.ops_key}, u:{settings.uid}:profile   Files: {settings.data_dir}",
          file=sys.stderr)
    return 1 if failed else 0


def _trace(event: dict, write) -> None:
    if event["type"] == "tool_call":
        args = json.dumps(event["args"], ensure_ascii=False)
        write(f"  . {event['name']} {args[:110]}")
    elif event["type"] == "tool_result" and not event["ok"] and not event.get("needs_approval"):
        write(f"    ! {str(event['error'])[:140]}")


def converse(agent, text: str, input_fn, write) -> AgentRun:
    """One user turn. Risky actions pause the agent; the human decides here, in the terminal (never the model)."""
    run = agent.ask(text, on_event=lambda e: _trace(e, write))
    while run.status == "needs_approval":
        ap = run.pending[0]
        answer = input_fn(f"\nThe agent wants to: {ap['reason']}\n  ({ap['tool']} {json.dumps(ap['args'])[:100]})\n"
                          "Allow this once? [y/N] ").strip().lower()
        decide = agent.approve if answer in ("y", "yes") else agent.deny
        run = decide(ap["id"], on_event=lambda e: _trace(e, write))
    return run


def chat(args: argparse.Namespace, input_fn=None, write=print) -> int:
    input_fn = input_fn or builtins.input
    rt = build_runtime(args, quiet=True)
    if rt is None:
        return 2
    settings, store, embedder, llm, blobs = rt.settings, rt.store, rt.embedder, rt.llm, rt.blobs
    try:
        orch = Orchestrator(settings, store, embedder, llm, blobs, _events(args))
        apply_policy_flags(orch.profile, args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    site = orch.register_site(args.site, args.device)
    use_agent = can_call_tools(llm) and not args.no_agent
    agent = ExecutionAgent(settings, store, embedder, llm, blobs, site, orch.profile) if use_agent else None
    if agent is not None:
        agent.ctx.extra["orchestrator"] = orch            # lets the agent queue background tasks (submit_task)
    if not use_agent:
        print("No LLM agent: " + ("--no-agent given" if args.no_agent else "no LLM configured (set LLM_API_KEY)") +
              ". Using the fixed request pipeline (explore / open a screen / make a demo / recommend).",
              file=sys.stderr)
    ops = Ops(store, settings.uid, site.domain)
    write(f"Talking about {site.domain}. Type /quit to leave.")
    try:
        while True:
            try:
                line = input_fn("\nyou> ").strip()
            except EOFError:
                break
            if not line:
                continue
            if line in ("/quit", "/exit"):
                break
            if agent is None:
                task = ops.create_task("chat", line, device=args.device)
                write(json.dumps(orch.run_task(task, line, site), indent=2, default=str))
                continue
            run = converse(agent, line, input_fn, write)
            write(f"\n{run.answer or '(no answer: ' + run.status + ')'}")
            write(f"  [{run.status}: {run.steps} model turns, {run.tokens} tokens, {len(run.tool_calls)} tool calls]")
    finally:
        if agent:
            agent.close()
    return 0


def serve_mcp(args: argparse.Namespace) -> int:
    """stdout belongs to the MCP protocol: everything human-readable goes to stderr."""
    import asyncio

    from .mcp_server import build_server, serve_stdio
    from .tools import ToolContext, build_toolbox

    rt = build_runtime(args, quiet=True)
    if rt is None:
        return 2
    settings, store, embedder, llm, blobs = rt.settings, rt.store, rt.embedder, rt.llm, rt.blobs
    orch = Orchestrator(settings, store, embedder, llm, blobs, NullEvents())
    site = orch.register_site(args.site, args.device)
    ctx = ToolContext(settings=settings, store=store, key=site.key, domain=site.domain, embedder=embedder, llm=llm,
                      policy=orch.profile.policy_for(site.domain, args.device), profile=orch.profile, device=args.device, blobs=blobs)
    ctx.extra["orchestrator"] = orch
    server = build_server(build_toolbox(ctx))
    print(f"SiteGraph MCP server for {site.domain} ready (stdio).", file=sys.stderr)
    try:
        asyncio.run(serve_stdio(server))
    finally:
        server.close_tools()
    return 0


def parse_variants(specs: list[str]) -> dict[str, dict]:
    """["brute:exploration_mode=brute,max_replans=1", "plain"] -> {"brute": {...}, "plain": {}}; values stay strings
    (the runner coerces them to each setting's type)."""
    out: dict[str, dict] = {}
    for spec in specs or []:
        name, _, rest = spec.partition(":")
        if not name.strip():
            raise ValueError(f"variant {spec!r} has no name (use NAME:key=value,...)")
        pairs = {}
        for part in filter(None, (x.strip() for x in rest.split(","))):
            k, eq, v = part.partition("=")
            if not eq or not k.strip():
                raise ValueError(f"variant {name!r}: {part!r} is not key=value")
            pairs[k.strip()] = v.strip()
        out[name.strip()] = pairs
    return out or {"default": {}}


def run_eval(args: argparse.Namespace) -> int:
    from .eval.report import load_runs, summarize, write_reports
    from .eval.runner import run_suite
    from .eval.suite import load_suite

    settings = load_settings(args.config)
    try:
        suite = load_suite(args.suite, args.site_url)
        variants = parse_variants(args.variant) if args.variant else (suite.variants or {"default": {}})
        baseline = load_runs(args.compare) if args.compare else None
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    try:
        GraphStore(settings.falkor_host, settings.falkor_port).redis.ping()
    except Exception as exc:
        print(f"Cannot reach FalkorDB at {settings.falkor_host}:{settings.falkor_port} ({exc}).\n"
              "Start it with: docker compose up -d", file=sys.stderr)
        return 2
    embedder = (CachedEmbedder(BgeEmbedder(settings.embed_model), settings.data_dir / "embed_cache.sqlite")
                if args.embedder == "bge" else HashEmbedder(settings.embed_dim))
    out = Path(args.out) if args.out else Path(settings.data_dir) / "eval" / time.strftime("%Y%m%d-%H%M%S")

    def progress(r) -> None:
        why = f" - {(r.error or '; '.join(r.failed))[:140]}" if r.outcome != "pass" else ""
        print(f"{r.outcome:5} {r.task_id} [{r.variant} #{r.rep}] {r.seconds:.0f}s {r.tokens} tokens{why}",
              file=sys.stderr)

    print(f"Suite {suite.name}: {len(suite.tasks)} tasks x {len(variants)} variant(s) x {args.repeat}",
          file=sys.stderr)
    try:
        runs = run_suite(suite, variants, args.repeat, out, settings, args.keep, embedder=embedder,
                         events=ConsoleEvents() if args.verbose else None, on_run=progress)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    md, html = write_reports(runs, out, baseline)
    print(json.dumps(summarize(runs), indent=2))
    print(f"\nResults: {out / 'results.jsonl'}\nReport:  {md}\n         {html}", file=sys.stderr)
    return 1 if any(r.outcome in ("fail", "error") for r in runs) else 0


def _events(args: argparse.Namespace):
    return ConsoleEvents() if getattr(args, "verbose", 0) else NullEvents()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(getattr(args, "verbose", 0))
    if args.cmd == "serve":
        print("The web UI/API is not built yet (plan tasks 18-19). Use: python -m sitegraph run <site> \"<request>\"",
              file=sys.stderr)
        return 2
    if args.cmd == "chat":
        return chat(args)
    if args.cmd == "mcp":
        return serve_mcp(args)
    if args.cmd == "eval":
        return run_eval(args)
    return run(args)
