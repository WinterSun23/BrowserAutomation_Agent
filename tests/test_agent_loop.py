from sitegraph.agents.loop import AgentLoop
from sitegraph.config import Settings
from sitegraph.llm import ChatMessage, ScriptedLLM, ToolCall
from sitegraph.tools.core import ToolContext, ToolSpec, Toolbox


def _box():
    box = Toolbox(ToolContext(settings=Settings()))
    box.register(ToolSpec("echo", "Echo", {"type": "object", "properties": {"msg": {"type": "string"}},
                                           "required": ["msg"]}, lambda c, a: {"said": a["msg"]}, "read"))
    box.register(ToolSpec("type_text", "Type text", {"type": "object", "properties": {"value": {"type": "string"}},
                                                     "required": ["value"]}, lambda c, a: {"typed": True}, "act",
                          sensitive_args=("value",)))
    box.register(ToolSpec("wipe", "Wipe everything", {"type": "object", "properties": {}},
                          lambda c, a: {"wiped": True}, "risky"))
    return box


def call(name, args=None, id="c1"):
    return ChatMessage("assistant", tool_calls=[ToolCall(id, name, args or {})])


def say(text, usage=None):
    return ChatMessage("assistant", content=text, usage=usage)


def test_loop_executes_tool_calls_then_answers():
    seen = []
    llm = ScriptedLLM([call("echo", {"msg": "hi"}), lambda m, t: seen.append(m[-1]) or say("it said hi")])
    run = AgentLoop(llm, _box(), "sys").run("say hi")
    assert run.status == "done" and run.answer == "it said hi" and run.steps == 2
    assert run.tool_calls == [{"name": "echo", "args": {"msg": "hi"}, "ok": True, "error": None}]
    assert seen[0].role == "tool" and seen[0].tool_call_id == "c1" and '"said": "hi"' in seen[0].content
    assert llm.calls[0][0][0].role == "system" and llm.calls[0][0][1].content == "say hi"
    assert {"echo", "wipe"} <= {t["function"]["name"] for t in llm.calls[0][1]}        # tools are offered


def test_unknown_tool_and_bad_arguments_are_fed_back_and_the_loop_continues():
    seen = []
    bad = ChatMessage("assistant", tool_calls=[ToolCall("c2", "echo", {}, arguments_error="arguments are not valid JSON")])
    llm = ScriptedLLM([call("teleport"), bad, lambda m, t: seen.extend(m[-3:]) or say("ok, giving up")])
    run = AgentLoop(llm, _box(), "sys").run("do it")
    assert run.status == "done" and run.answer == "ok, giving up"
    texts = " ".join(m.content or "" for m in seen if m.role == "tool")
    assert "unknown tool 'teleport'" in texts and "not valid JSON" in texts


def test_max_steps_stops_a_looping_model():
    llm = ScriptedLLM([call("echo", {"msg": "again"}, id=f"c{i}") for i in range(10)])
    run = AgentLoop(llm, _box(), "sys", max_steps=4).run("loop forever")
    assert run.status == "max_steps" and run.steps == 4 and len(llm.calls) == 4


def test_token_budget_stops_the_run():
    llm = ScriptedLLM([ChatMessage("assistant", tool_calls=[ToolCall(f"c{i}", "echo", {"msg": "x"})],
                                   usage={"total_tokens": 100}) for i in range(5)])
    run = AgentLoop(llm, _box(), "sys", max_tokens=250).run("go")
    assert run.status == "max_tokens" and run.tokens == 300 and run.steps == 3


def test_llm_failure_becomes_an_error_status():
    run = AgentLoop(ScriptedLLM([]), _box(), "sys").run("hello")
    assert run.status == "error" and "exhausted" in run.answer


def test_needs_approval_pauses_and_the_run_resumes_after_approve():
    box = _box()
    llm = ScriptedLLM([call("wipe"), call("wipe", id="c2"), say("wiped, as approved")])
    history: list = []
    loop = AgentLoop(llm, box, "sys")
    paused = loop.run("wipe it", history)
    assert paused.status == "needs_approval" and paused.pending[0]["tool"] == "wipe"
    assert history[-1].role == "tool"                                                  # every tool call got an answer
    box.ctx.approvals.approve(paused.pending[0]["id"])
    resumed = loop.run("The user approved. You may retry the action.", history)
    assert resumed.status == "done" and resumed.answer == "wiped, as approved"
    assert [c["name"] for c in resumed.tool_calls] == ["wipe"] and resumed.tool_calls[0]["ok"]


def test_denied_action_is_reported_to_the_model():
    box = _box()
    llm = ScriptedLLM([call("wipe"), call("wipe", id="c2"), say("understood, not doing it")])
    history: list = []
    loop = AgentLoop(llm, box, "sys")
    box.ctx.approvals.deny(loop.run("wipe it", history).pending[0]["id"])
    run = loop.run("The user denied it.", history)
    assert run.status == "done" and "denied" in history[-2].content


def test_sensitive_values_are_redacted_in_events_and_the_run_record():
    events = []
    llm = ScriptedLLM([call("type_text", {"value": "hunter-secret-42"}), say("typed")])
    run = AgentLoop(llm, _box(), "sys").run("type it", on_event=events.append)
    assert run.status == "done"
    assert run.tool_calls[0]["args"] == {"value": "<hidden>"}
    assert "hunter-secret-42" not in repr(events) and "hunter-secret-42" not in repr(run.tool_calls)


def test_events_describe_each_step():
    events = []
    llm = ScriptedLLM([call("echo", {"msg": "e"}), say("fin")])
    AgentLoop(llm, _box(), "sys").run("go", on_event=events.append)
    assert [e["type"] for e in events] == ["tool_call", "tool_result", "assistant"]


def test_a_tool_can_show_the_model_an_image_on_its_next_turn():
    from sitegraph.tools.core import ToolResult
    box = _box()
    box.register(ToolSpec("preview", "Preview", {"type": "object", "properties": {}},
                          lambda c, a: ToolResult(True, data={"shot": 1}, image_png=b"\x89PNGdata", image_label="shot 1"),
                          "read"))
    llm = ScriptedLLM([call("preview"), say("looks fine")])
    run = AgentLoop(llm, box, "sys").run("check it", image_png=b"\x89PNGfirst")
    assert run.status == "done"
    assert llm.images == [b"\x89PNGfirst", b"\x89PNGdata"]                      # caller's image first, then the preview
    assert any(m.role == "user" and "shot 1" in (m.content or "") for m in llm.calls[1][0])
    assert '"shot": 1' in llm.calls[1][0][-2].content                          # the tool message itself stays text


def test_a_tool_can_ask_the_user_a_question_and_end_the_turn():
    from sitegraph.tools.core import ToolResult
    box = _box()
    box.register(ToolSpec("ask", "Ask", {"type": "object", "properties": {}},
                          lambda c, a: ToolResult(True, data={"asked": True}, pause="Which format do you want?"), "read"))
    llm = ScriptedLLM([call("ask"), say("never reached in this turn")])
    history: list = []
    run = AgentLoop(llm, box, "sys").run("make a demo", history)
    assert run.status == "needs_input" and run.answer == "Which format do you want?" and run.steps == 1
    assert history[-1].role == "tool" and len(llm.calls) == 1                  # the tool call was answered; no second model call


def test_a_call_repeated_with_the_same_result_is_warned_about_then_stops_the_run():
    seen = []
    llm = ScriptedLLM([lambda m, t, i=i: seen.append(m[-1].content) or call("echo", {"msg": "x"}, id=f"c{i}")
                       for i in range(8)])
    run = AgentLoop(llm, _box(), "sys", repeat_limit=5).run("go")
    assert run.status == "stuck" and run.steps == 5 and len(run.tool_calls) == 5
    assert "echo" in run.answer and "5 times" in run.answer
    assert "repeated" not in seen[2] and "repeated" in seen[3] and "repeated" in seen[4]      # warned from the 3rd repeat


def test_different_arguments_or_results_are_not_repeats():
    llm = ScriptedLLM([call("echo", {"msg": f"m{i}"}, id=f"c{i}") for i in range(8)] + [say("done")])
    run = AgentLoop(llm, _box(), "sys", repeat_limit=3).run("go")
    assert run.status == "done" and len(run.tool_calls) == 8


def _observing_box():
    n = {"i": 0}

    def observe(ctx, args):
        n["i"] += 1
        return {"page": f"page-{n['i']}"}

    box = Toolbox(ToolContext(settings=Settings()))
    box.register(ToolSpec("observe", "Look", {"type": "object", "properties": {}}, observe, "read"))
    return box


def test_only_the_latest_observations_are_sent_to_the_model_again():
    sent = []
    steps = [lambda m, t, i=i: sent.append([x.content for x in m if x.role == "tool"]) or call("observe", id=f"c{i}")
             for i in range(5)] + [lambda m, t: sent.append([x.content for x in m if x.role == "tool"]) or say("ok")]
    llm = ScriptedLLM(steps)
    history: list = []
    run = AgentLoop(llm, _observing_box(), "sys", keep_observations=2).run("look around", history)
    assert run.status == "done"
    last = sent[-1]
    assert len(last) == 5 and [("removed" in c) for c in last] == [True, True, True, False, False]
    assert "page-4" in last[3] and "page-5" in last[4]
    assert all("page-" in m.content for m in history if m.role == "tool")              # the stored history is complete
