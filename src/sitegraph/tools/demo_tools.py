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
    _demo(ctx).set_caption(int(args["shot_id"]), args["caption"], args.get("narration"))
    return {"shot_id": args["shot_id"], "caption": args["caption"][:120]}


def list_shots(ctx: ToolContext, args: dict):
    return _demo(ctx).list_shots()


def compose_storyboard(ctx: ToolContext, args: dict):
    return _demo(ctx).compose([int(i) for i in args["shot_ids"]], args["title"])


def render(ctx: ToolContext, args: dict):
    return _demo(ctx).render(list(args["formats"]), bool(args.get("narration")))


def rehearsal_log(ctx: ToolContext, args: dict):
    return {"actions": _demo(ctx).rehearsal_log()}


def write_script(ctx: ToolContext, args: dict):
    return _demo(ctx).write_script(args["title"], args.get("intro", ""), args.get("outro", ""), list(args["scenes"]),
                                   args.get("goal"))


def record_final(ctx: ToolContext, args: dict):
    return _demo(ctx).record_final()


def review_take(ctx: ToolContext, args: dict):
    png, label = _demo(ctx).review(int(args["scene"]))
    return ToolResult(True, data={"scene": args["scene"], "note": "the image is attached to your next turn"},
                      image_png=png, image_label=label)


def register(box: Toolbox) -> None:
    sid = {"shot_id": {"type": "integer"}}
    box.register(ToolSpec("rehearsal_log", "List the actions that worked during the rehearsal, with their ids "
                          "(a1, a2, ...) for write_script.", {"type": "object", "properties": {}}, rehearsal_log, "read"))
    scene = {"type": "object", "properties": {
        "action": {"type": "string", "description": "a rehearsal action id, or omit for a scene that explains the page"},
        "caption": {"type": "string", "description": "explains the step for a newcomer: why, and what to notice"},
        "narration": {"type": "string"}, "hold": {"type": "number", "description": "seconds on screen (2.5-8)"},
        "zoom": {"type": "boolean"}}, "required": ["caption"]}
    box.register(ToolSpec(
        "write_script", "Write the demo script from the rehearsal: title, intro, outro and scenes in order (only the "
        "actions that worked, no detours). The user may have to approve it before recording.",
        {"type": "object", "properties": {"title": {"type": "string"}, "goal": {"type": "string"},
                                          "intro": {"type": "string"}, "outro": {"type": "string"},
                                          "scenes": {"type": "array", "items": scene, "minItems": 1}},
         "required": ["title", "scenes"]}, write_script, "act"))
    box.register(ToolSpec(
        "record_final", "Record the approved script as ONE continuous take in a fresh browser (title card, scenes, "
        "closing card). A failure names the scene to fix.", {"type": "object", "properties": {}}, record_final, "act",
        untrusted_output=True))
    box.register(ToolSpec(
        "review_take", "Look at one scene of the final take (0 = title card, last = closing card); the image is shown "
        "to you on your next turn.", {"type": "object", "properties": {"scene": {"type": "integer"}},
                                       "required": ["scene"]}, review_take, "read"))
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
                          {"type": "object", "properties": {**sid, "caption": {"type": "string"},
                                                          "narration": {"type": "string"}},
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
            "formats": {"type": "array", "items": {"type": "string"}},
            "narration": {"type": "boolean"}}, "required": ["formats"]}, render, "act"))
