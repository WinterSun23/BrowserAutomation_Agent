from __future__ import annotations

from .core import ApprovalGate, ToolContext, ToolResult, ToolSpec, Toolbox


def build_toolbox(ctx: ToolContext, include: set[str] | None = None) -> Toolbox:
    """A Toolbox with every tool registered (graph, browser, engine, memory), or only those named in `include`."""
    from . import (browser_tools, dialog_tools, engine_tools, git_tools, graph_tools, insight_tools, ops_tools,
                   profile_tools)

    box = Toolbox(ctx)
    for module in (graph_tools, browser_tools, engine_tools, dialog_tools, profile_tools, git_tools, insight_tools,
                   ops_tools):
        module.register(box)
    if include is not None:
        box.restrict(include)
    return box


__all__ = ["ApprovalGate", "ToolContext", "ToolResult", "ToolSpec", "Toolbox", "build_toolbox"]
