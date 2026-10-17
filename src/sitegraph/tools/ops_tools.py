"""Tools that hand work to the Orchestrator and follow it: submit a request as a background task and read its status."""
from __future__ import annotations

from ..ops import Ops
from .core import ToolContext, ToolSpec, Toolbox


def _orch(ctx: ToolContext):
    orch = ctx.extra.get("orchestrator")
    if orch is None:
        raise ValueError("no orchestrator is attached to this session")
    return orch


def submit_task(ctx: ToolContext, args: dict):
    return {"task": _orch(ctx).submit(args["text"], ctx.domain, ctx.device),
            "note": "running in the background; poll task_status with this id"}


def task_status(ctx: ToolContext, args: dict):
    if not ctx.domain:
        raise ValueError("no site is selected")
    return Ops(ctx.store, ctx.settings.uid, ctx.domain).get_task(args["task_id"])


def register(box: Toolbox) -> None:
    box.register(ToolSpec(
        "submit_task", "Queue a plain-language request (explore the site, open a screen, make a demo...) as a "
        "background task and return its id. Use for long jobs; risky steps are still gated.",
        {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}, submit_task, "act"))
    box.register(ToolSpec(
        "task_status", "Status and result of a task started with submit_task.",
        {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]}, task_status, "read",
        untrusted_output=True))
