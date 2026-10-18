"""The LLM tool-calling loop shared by every agent. It owns no policy: tools enforce safety, the loop only bounds
the number of turns and tokens and feeds every result (including errors and approval requests) back to the model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..llm import ChatMessage, ToolLLM
from ..tools.core import ToolResult, Toolbox


@dataclass
class AgentRun:
    status: str                          # done | max_steps | max_tokens | needs_approval | needs_input | error
    answer: str | None = None
    steps: int = 0                       # model turns used
    tokens: int = 0
    tool_calls: list[dict] = field(default_factory=list)   # {name, args (sensitive values hidden), ok, error}
    pending: list[dict] = field(default_factory=list)      # approval requests waiting for a human
    messages: list[ChatMessage] = field(default_factory=list)


class AgentLoop:
    def __init__(self, llm: ToolLLM, toolbox: Toolbox, system_prompt: str, max_steps: int = 60,
                 max_tokens: int = 400000):
        self.llm, self.toolbox, self.system_prompt = llm, toolbox, system_prompt
        self.max_steps, self.max_tokens = max_steps, max_tokens

    def run(self, user_text: str | None, history: list[ChatMessage] | None = None,
            on_event: Callable[[dict], None] | None = None, image_png: bytes | None = None) -> AgentRun:
        """Add `user_text` to the conversation (`history`, extended in place) and let the model work until it answers.
        Call again with the same history after approving or denying a paused action."""
        history = history if history is not None else []
        if user_text:
            history.append(ChatMessage("user", content=user_text))
        emit = on_event or (lambda e: None)
        run = AgentRun("max_steps", messages=history)
        max_chars = self.toolbox.ctx.settings.tool_result_max_chars
        next_image = image_png            # a screenshot for the model's next turn (first turn: the caller's, later: previews)
        while run.steps < self.max_steps:
            try:
                reply = self.llm.chat([ChatMessage("system", content=self.system_prompt)] + history,
                                      self.toolbox.specs(), image_png=next_image)
            except Exception as exc:
                run.status, run.answer = "error", f"The model call failed: {exc}"
                return run
            next_image = None
            run.steps += 1
            run.tokens += (reply.usage or {}).get("total_tokens", 0)
            history.append(reply)
            if not reply.tool_calls:
                run.status, run.answer = "done", reply.content
                emit({"type": "assistant", "content": reply.content})
                return run
            question, previews = None, []
            for tc in reply.tool_calls:
                shown = self.toolbox.redacted(tc.name, tc.arguments)
                emit({"type": "tool_call", "name": tc.name, "args": shown})
                if tc.arguments_error:
                    result = ToolResult(False, error=f"could not read the arguments for {tc.name}: {tc.arguments_error}")
                else:
                    result = self.toolbox.call(tc.name, tc.arguments)
                run.tool_calls.append({"name": tc.name, "args": shown, "ok": result.ok, "error": result.error})
                emit({"type": "tool_result", "name": tc.name, "ok": result.ok, "error": result.error,
                      "needs_approval": result.needs_approval})
                history.append(ChatMessage("tool", content=result.to_message_text(max_chars),
                                           tool_call_id=tc.id, name=tc.name))
                if result.needs_approval:
                    run.pending.append(result.needs_approval)
                if result.pause:
                    question = result.pause
                if result.image_png:
                    previews.append(result)
            if run.pending:                       # every tool call above has an answer; hand over to a human
                run.status = "needs_approval"
                return run
            if question:                          # the agent asked the user something: its turn ends here
                run.status, run.answer = "needs_input", question
                emit({"type": "assistant", "content": question})
                return run
            if previews:                          # show the model the images its tools produced, on its next turn
                labels = [p.image_label or "preview" for p in previews]
                history.append(ChatMessage("user", content="[Attached image: " + "; ".join(labels) + "]"))
                next_image = [(label, p.image_png) for label, p in zip(labels, previews)]
            if run.tokens >= self.max_tokens:
                run.status = "max_tokens"
                run.answer = "Stopped: the token budget for this request was used up."
                return run
        run.answer = "Stopped: the step limit for this request was reached."
        return run
