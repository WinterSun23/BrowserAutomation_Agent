"""Tools over the version history of the site map: look at it (log, blame), mark a version (tag), undo a change
(revert, risky) and re-record a saved demo (against the current map or the map it was made with)."""
from __future__ import annotations

from ..demo.director import DemoDirector
from ..git import GitOps
from .core import ToolContext, ToolSpec, Toolbox


def _git(ctx: ToolContext) -> GitOps:
    if not ctx.domain:
        raise ValueError("no site is selected")
    return GitOps(ctx.store, ctx.settings.uid, ctx.domain, ctx.settings)


def git_log(ctx: ToolContext, args: dict):
    return _git(ctx).log(ctx.device, int(args.get("limit", 20)))


def blame(ctx: ToolContext, args: dict):
    return _git(ctx).blame(args["gid"])


def git_tag(ctx: ToolContext, args: dict):
    return {"graph": _git(ctx).tag(ctx.device, args["name"])}


def git_revert(ctx: ToolContext, args: dict):
    return {"revert_commit": _git(ctx).revert(ctx.device, args["commit"])}


def rerender_demo(ctx: ToolContext, args: dict):
    d = DemoDirector(ctx.store, ctx.key, ctx.domain, ctx.settings, ctx.embedder, ctx.llm, ctx.blobs, ctx.profile,
                     ctx.device, git=_git(ctx))
    res = d.rerender(args["storyboard"], pinned=bool(args.get("pinned")))
    return {"storyboard": res.storyboard_gid, "renders": [{"format": f, "file": ref} for f, ref in res.renders]}


def _obj(required=(), **props) -> dict:
    return {"type": "object", "properties": props, "required": list(required)}


_S = {"type": "string"}


def register(box: Toolbox) -> None:
    box.register(ToolSpec("git_log", "The version history of this site's map: commits, who made them, tags.",
                          _obj(limit={"type": "integer", "minimum": 1, "maximum": 100}), git_log, "read"))
    box.register(ToolSpec("blame", "Which commits and agents touched a state, element or action (use its gid).",
                          _obj(["gid"], gid=_S), blame, "read"))
    box.register(ToolSpec("git_tag", "Save a named snapshot of the current map.", _obj(["name"], name=_S), git_tag,
                          "act"))
    box.register(ToolSpec("git_revert", "Undo one commit of the map (history is kept; the undo is itself a commit). "
                          "Only on the user's request.", _obj(["commit"], commit=_S), git_revert, "risky"))
    box.register(ToolSpec("rerender_demo", "Record a saved demo (storyboard gid) again. pinned=true uses the map as it "
                          "was when the demo was made; otherwise the current map.",
                          _obj(["storyboard"], storyboard=_S, pinned={"type": "boolean"}), rerender_demo, "act"))
