"""Talking back to the user."""
from __future__ import annotations

from .core import ToolContext, ToolResult, ToolSpec, Toolbox


def ask_user(ctx: ToolContext, args: dict):
    # the agent loop ends the turn with status 'needs_input'; the user's next message continues the conversation
    return ToolResult(True, data={"asked": args["question"]}, pause=args["question"])


def register(box: Toolbox) -> None:
    box.register(ToolSpec(
        "ask_user", "Ask the user a short clarifying question when the request is ambiguous or a choice is theirs "
        "(for example which format or which part of the site). Your turn ends and resumes with their answer. Do not use "
        "it for things you can find out with the other tools.",
        {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}, ask_user, "read"))
