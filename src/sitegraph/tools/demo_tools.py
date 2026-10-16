"""Tools of the Demo agent. They are thin: the recording state lives in the DemoAgent (ctx.extra['demo'])."""
from __future__ import annotations

from .browser_tools import _element, _guard
from .core import ToolContext, ToolResult, ToolSpec, Toolbox


def _demo(ctx: ToolContext):
    demo = ctx.extra.get("demo")
    if demo is None:
        raise ValueError("no demo is being recorded")
    return demo


def capture_route(ctx: ToolContext, args: dict):
    route = ctx.extra.get("routes", {}).get(args["route_id"])
    if route is None:
        raise ValueError(f"unknown route_id '{args['route_id']}': call plan_route first")
    shots = _demo(ctx).capture_route(route, bool(args.get("visible", True)))
    ctx.extra.pop("elements", None)
    return {"shots": shots}


def capture_interaction(ctx: ToolContext, args: dict):
    el = _element(ctx, args["element_id"])
    _guard(ctx, el)                                  # same risk gate as every other browser action
    shot = _demo(ctx).capture_interaction(el, args.get("kind", "click"), args["caption"])
    ctx.extra.pop("elements", None)
    return shot


def capture_still(ctx: ToolContext, args: dict):
    return _demo(ctx).capture_still(args["caption"])


def preview_frame(ctx: ToolContext, args: dict):
    png, label = _demo(ctx).keyframe(int(args["shot_id"]))
    return ToolResult(True, data={"shot_id": args["shot_id"], "note": "the image is attached to your next turn"},
                      image_png=png, image_label=label)


def discard_shot(ctx: ToolContext, args: dict):
    _demo(ctx).discard(int(args["shot_id"]))
    return {"discarded": args["shot_id"]}


def set_caption(ctx: ToolContext, args: dict):
    _demo(ctx).set_caption(int(args["shot_id"]), args["caption"])
    return {"shot_id": args["shot_id"], "caption": args["caption"][:120]}


def list_shots(ctx: ToolContext, args: dict):
    return _demo(ctx).list_shots()


def compose_storyboard(ctx: ToolContext, args: dict):
    return _demo(ctx).compose([int(i) for i in args["shot_ids"]], args["title"])


def render(ctx: ToolContext, args: dict):
    return _demo(ctx).render(list(args["formats"]))


def register(box: Toolbox) -> None:
    sid = {"shot_id": {"type": "integer"}}
    box.register(ToolSpec(
        "capture_route", "Record a route from plan_route as shots (one per step, each highlighting the element). "
        "visible=false only drives to the start of the demo without recording.",
        {"type": "object", "properties": {"route_id": {"type": "string"}, "visible": {"type": "boolean"}},
         "required": ["route_id"]}, capture_route, "act", untrusted_output=True))
    box.register(ToolSpec(
        "capture_interaction", "Record one interaction on the current page (click or hover an element from observe), "
        "highlighted, with a caption.", {"type": "object", "properties": {
            "element_id": {"type": "string"}, "kind": {"type": "string", "enum": ["click", "hover"]},
            "caption": {"type": "string"}}, "required": ["element_id", "caption"]},
        capture_interaction, "act", untrusted_output=True))
    box.register(ToolSpec("capture_still", "Record the current page as a still shot with a caption.",
                          {"type": "object", "properties": {"caption": {"type": "string"}}, "required": ["caption"]},
                          capture_still, "act"))
    box.register(ToolSpec("preview_frame", "Look at the keyframe of a recorded shot (the image is shown to you on your "
                          "next turn) to check the highlight, the caption and that nothing covers the page.",
                          {"type": "object", "properties": sid, "required": ["shot_id"]}, preview_frame, "read"))
    box.register(ToolSpec("discard_shot", "Drop a bad shot from the demo (then capture it again).",
                          {"type": "object", "properties": sid, "required": ["shot_id"]}, discard_shot, "act"))
    box.register(ToolSpec("set_caption", "Change a shot's on-screen caption (12 words or fewer).",
                          {"type": "object", "properties": {**sid, "caption": {"type": "string"}},
                           "required": ["shot_id", "caption"]}, set_caption, "act"))
    box.register(ToolSpec("list_shots", "List the shots recorded so far.", {"type": "object", "properties": {}},
                          list_shots, "read"))
    box.register(ToolSpec(
        "compose_storyboard", "Choose and order the shots that make up the demo and save the storyboard.",
        {"type": "object", "properties": {"shot_ids": {"type": "array", "items": {"type": "integer"}},
                                          "title": {"type": "string"}}, "required": ["shot_ids", "title"]},
        compose_storyboard, "act"))
    box.register(ToolSpec(
        "render", "FINAL STEP: stop recording and render the storyboard in the given formats "
        "(video, gif, screenshot, guide, clickthrough).", {"type": "object", "properties": {
            "formats": {"type": "array", "items": {"type": "string"}}}, "required": ["formats"]}, render, "act"))
