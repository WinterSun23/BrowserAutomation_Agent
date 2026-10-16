from __future__ import annotations

import argparse
import builtins
import json
import sys

from .agents.execution_agent import ExecutionAgent
from .agents.loop import AgentRun
from .agents.orchestrator import Orchestrator
from .blobs import BlobStore
from .config import Settings, load_settings
from .db import GraphStore
from .embed import BgeEmbedder, HashEmbedder
from .events import NullEvents
from .llm import LiteLLMClient, OfflineLLM
from .ops import Ops


def make_llm(settings: Settings):
    """A real model when LLM_API_KEY (or LLM_API_BASE for a local server) is set, otherwise the offline fallback."""
    if settings.llm_api_key or settings.llm_api_base:
        return LiteLLMClient(settings.llm_model, settings.llm_api_base, settings.llm_api_key)
    return OfflineLLM()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m sitegraph", description="SiteGraph: explore, automate and demo a website.")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="register a site and run one or more requests against it, in order")
    r.add_argument("site", help="entry URL of a public website, e.g. https://books.toscrape.com/")
    r.add_argument("requests", nargs="+", help='e.g. "explore the site" "open cart" "make a gif of the menu"')
    r.add_argument("--device", default="desktop", choices=["desktop", "tablet", "phone"])
    r.add_argument("--navigators", type=int, help="how many agents explore in parallel (default: config.yaml)")
    r.add_argument("--embedder", default="bge", choices=["bge", "hash"],
                   help="bge = real semantic model (downloads ~130 MB once); hash = offline, weak, for quick tests")
    r.add_argument("--config", default="config.yaml")
    r.add_argument("--uid", help="user id that namespaces the graphs (default: config.yaml)")
    c = sub.add_parser("chat", help="talk to an agent that explores, operates and demos the site with tools")
    c.add_argument("site", help="entry URL of a public website")
    c.add_argument("--device", default="desktop", choices=["desktop", "tablet", "phone"])
    c.add_argument("--embedder", default="bge", choices=["bge", "hash"])
    c.add_argument("--config", default="config.yaml")
    c.add_argument("--uid", help="user id that namespaces the graphs (default: config.yaml)")
    c.add_argument("--no-agent", action="store_true", help="skip the LLM agent and use the fixed request pipeline")
    m = sub.add_parser("mcp", help="serve the tools over MCP (stdio) so an MCP client such as Claude Desktop can use them")
    m.add_argument("site", help="entry URL of a public website")
    m.add_argument("--device", default="desktop", choices=["desktop", "tablet", "phone"])
    m.add_argument("--embedder", default="bge", choices=["bge", "hash"])
    m.add_argument("--config", default="config.yaml")
    m.add_argument("--uid", help="user id that namespaces the graphs (default: config.yaml)")
    sub.add_parser("serve", help="web UI (not built yet)")
    return p


def run(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    if args.uid:
        settings.uid = args.uid
    try:
        store = GraphStore(settings.falkor_host, settings.falkor_port)
        store.redis.ping()
    except Exception as exc:
        print(f"Cannot reach FalkorDB at {settings.falkor_host}:{settings.falkor_port} ({exc}).\n"
              "Start it with: docker compose up -d", file=sys.stderr)
        return 2
    llm = make_llm(settings)
    if isinstance(llm, OfflineLLM):
        print("No LLM configured: using built-in defaults (keyword routing, plain summaries and captions). "
              "Set LLM_API_KEY to enable a model.", file=sys.stderr)
    if args.embedder == "bge":
        print(f"Loading embedding model {settings.embed_model} (downloads on first use)...", file=sys.stderr)
        embedder = BgeEmbedder(settings.embed_model)
    else:
        embedder = HashEmbedder(settings.embed_dim)
    orch = Orchestrator(settings, store, embedder, llm, BlobStore(settings.blob_dir), NullEvents())
    site = orch.register_site(args.site, args.device)
    ops = Ops(store, settings.uid, site.domain)
    failed = False
    for text in args.requests:
        print(f"\n== {text}", file=sys.stderr)
        task = ops.create_task("chat", text, device=args.device)
        result = orch.run_task(task, text, site, args.navigators)
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
    settings = load_settings(args.config)
    if args.uid:
        settings.uid = args.uid
    try:
        store = GraphStore(settings.falkor_host, settings.falkor_port)
        store.redis.ping()
    except Exception as exc:
        print(f"Cannot reach FalkorDB at {settings.falkor_host}:{settings.falkor_port} ({exc}).\n"
              "Start it with: docker compose up -d", file=sys.stderr)
        return 2
    llm = make_llm(settings)
    embedder = (BgeEmbedder(settings.embed_model) if args.embedder == "bge" else HashEmbedder(settings.embed_dim))
    blobs = BlobStore(settings.blob_dir)
    orch = Orchestrator(settings, store, embedder, llm, blobs, NullEvents())
    site = orch.register_site(args.site, args.device)
    use_agent = not isinstance(llm, OfflineLLM) and not args.no_agent
    agent = ExecutionAgent(settings, store, embedder, llm, blobs, site, orch.profile) if use_agent else None
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

    settings = load_settings(args.config)
    if args.uid:
        settings.uid = args.uid
    try:
        store = GraphStore(settings.falkor_host, settings.falkor_port)
        store.redis.ping()
    except Exception as exc:
        print(f"Cannot reach FalkorDB at {settings.falkor_host}:{settings.falkor_port} ({exc}).", file=sys.stderr)
        return 2
    llm = make_llm(settings)
    embedder = BgeEmbedder(settings.embed_model) if args.embedder == "bge" else HashEmbedder(settings.embed_dim)
    blobs = BlobStore(settings.blob_dir)
    orch = Orchestrator(settings, store, embedder, llm, blobs, NullEvents())
    site = orch.register_site(args.site, args.device)
    ctx = ToolContext(settings=settings, store=store, key=site.key, domain=site.domain, embedder=embedder, llm=llm,
                      policy=orch.profile.default_policy(), profile=orch.profile, device=args.device, blobs=blobs)
    server = build_server(build_toolbox(ctx))
    print(f"SiteGraph MCP server for {site.domain} ready (stdio).", file=sys.stderr)
    try:
        asyncio.run(serve_stdio(server))
    finally:
        server.close_tools()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "serve":
        print("The web UI/API is not built yet (plan tasks 18-19). Use: python -m sitegraph run <site> \"<request>\"",
              file=sys.stderr)
        return 2
    if args.cmd == "chat":
        return chat(args)
    if args.cmd == "mcp":
        return serve_mcp(args)
    return run(args)
