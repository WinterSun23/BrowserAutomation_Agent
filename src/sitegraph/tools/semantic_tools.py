"""Tools over the cross-device semantic graph: rebuild it, report where devices differ, carry a workflow across."""
from __future__ import annotations

from ..semantic.distiller import Distiller, semantic_key
from ..semantic.uses import PortError, parity_report, port_workflow
from .core import ToolContext, ToolSpec, Toolbox


def semantic_rebuild(ctx: ToolContext, args: dict):
    if ctx.profile is None:
        raise ValueError("the user's profile is not available in this session")
    return Distiller(ctx.store, ctx.settings.uid, ctx.embedder, ctx.llm, ctx.settings, ctx.profile).rebuild()


def semantic_parity(ctx: ToolContext, args: dict):
    if not ctx.store.exists(semantic_key(ctx.settings.uid)):
        raise ValueError("the semantic graph has not been built yet: call semantic_rebuild first")
    return parity_report(ctx.store, ctx.settings.uid)


def port_workflow_tool(ctx: ToolContext, args: dict):
    try:
        wf = port_workflow(ctx.store, ctx.settings.uid, ctx.embedder, ctx.domain, args["workflow"],
                           args["from_device"], args["to_device"])
    except PortError as exc:
        raise ValueError(str(exc)) from exc
    return {"workflow": wf, "on_device": args["to_device"]}


_DEV = {"type": "string", "enum": ["desktop", "tablet", "phone"]}


def register(box: Toolbox) -> None:
    box.register(ToolSpec("semantic_rebuild", "Rebuild the cross-device semantic graph (shared UI patterns, the same "
                          "screen on each device, common flows) from all explored devices.",
                          {"type": "object", "properties": {}}, semantic_rebuild, "act"))
    box.register(ToolSpec("semantic_parity", "Where devices differ: screens missing on a device or needing more steps "
                          "than on desktop. Needs semantic_rebuild first.", {"type": "object", "properties": {}},
                          semantic_parity, "read", untrusted_output=True))
    box.register(ToolSpec("port_workflow", "Copy a saved workflow from one device to another using the shared "
                          "screens; fails with a reason when a screen has no counterpart yet.",
                          {"type": "object", "properties": {"workflow": {"type": "string"}, "from_device": _DEV,
                                                            "to_device": _DEV},
                           "required": ["workflow", "from_device", "to_device"]}, port_workflow_tool, "act"))
