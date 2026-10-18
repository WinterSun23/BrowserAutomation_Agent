import os
from types import SimpleNamespace

import pytest
from sitegraph.llm import (ChatMessage, LiteLLMClient, OfflineLLM, ScriptedLLM, ToolCall, from_litellm,
                           to_litellm)

TOOLS = [{"type": "function", "function": {"name": "click", "description": "Click an element",
                                           "parameters": {"type": "object", "properties": {"id": {"type": "string"}},
                                                          "required": ["id"]}}}]


def test_scripted_llm_replays_in_order_and_records_calls():
    first = ChatMessage("assistant", tool_calls=[ToolCall("c1", "click", {"id": "e1"})])
    second = ChatMessage("assistant", content="done")
    llm = ScriptedLLM([first, lambda messages, tools: second])
    user = [ChatMessage("user", content="go")]
    assert llm.chat(user, TOOLS) is first
    assert llm.chat(user + [first], TOOLS) is second
    assert [len(m) for m, _ in llm.calls] == [1, 2] and llm.calls[0][1] == TOOLS
    with pytest.raises(RuntimeError, match="exhausted"):
        llm.chat(user, TOOLS)


def test_chat_message_roundtrip_to_litellm_dict():
    msg = ChatMessage("assistant", tool_calls=[ToolCall("c1", "click", {"id": "e1", "n": 2})])
    d = to_litellm(msg)
    assert d["role"] == "assistant" and d["tool_calls"][0]["function"]["name"] == "click"
    assert d["tool_calls"][0]["type"] == "function" and d["tool_calls"][0]["id"] == "c1"
    assert d["tool_calls"][0]["function"]["arguments"] == '{"id": "e1", "n": 2}'     # a JSON string, as the API wants
    result = to_litellm(ChatMessage("tool", content="ok", tool_call_id="c1", name="click"))
    assert result == {"role": "tool", "content": "ok", "tool_call_id": "c1", "name": "click"}

    api = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(
        id="c9", function=SimpleNamespace(name="click", arguments='{"id": "e7"}'))])
    back = from_litellm(api)
    assert back.role == "assistant" and back.tool_calls == [ToolCall("c9", "click", {"id": "e7"})]


def test_bad_tool_arguments_become_empty_dict_with_an_error_marker():
    api = SimpleNamespace(content="", tool_calls=[
        SimpleNamespace(id="a", function=SimpleNamespace(name="click", arguments="{not json")),
        SimpleNamespace(id="b", function=SimpleNamespace(name="click", arguments='["a list"]')),
        SimpleNamespace(id="c", function=SimpleNamespace(name="click", arguments=None))])
    calls = from_litellm(api).tool_calls
    assert [c.arguments for c in calls] == [{}, {}, {}]
    assert calls[0].arguments_error and calls[1].arguments_error and calls[2].arguments_error is None


def test_litellm_client_chat_passes_tools_and_attaches_image(monkeypatch):
    seen = {}

    def fake_completion(**kw):
        seen.update(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hi", tool_calls=None))])

    import litellm
    monkeypatch.setattr(litellm, "completion", fake_completion)
    client = LiteLLMClient("openai/test", api_base="http://x/v1", api_key="k")
    out = client.chat([ChatMessage("system", content="s"), ChatMessage("user", content="look")], TOOLS,
                      image_png=b"\x89PNG0")
    assert out.content == "hi" and out.tool_calls is None
    assert seen["tools"] == TOOLS and seen["temperature"] == 0.0 and seen["api_base"] == "http://x/v1"
    last = seen["messages"][-1]
    assert last["role"] == "user" and last["content"][0] == {"type": "text", "text": "look"}
    assert last["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_litellm_client_chat_sends_several_labelled_images_in_order(monkeypatch):
    seen = {}

    def fake_completion(**kw):
        seen.update(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hi", tool_calls=None))])

    import litellm
    monkeypatch.setattr(litellm, "completion", fake_completion)
    client = LiteLLMClient("openai/test", api_base="http://x/v1", api_key="k")
    client.chat([ChatMessage("user", content="[Attached image: shot 1; shot 2]")], TOOLS,
                image_png=[("shot 1", b"PNG1"), ("shot 2", b"PNG2")])
    parts = seen["messages"][-1]["content"]
    assert [p["type"] for p in parts] == ["text", "text", "image_url", "text", "image_url"]
    assert parts[1]["text"] == "Image 1: shot 1" and parts[3]["text"] == "Image 2: shot 2"
    assert parts[2]["image_url"]["url"] != parts[4]["image_url"]["url"]


def test_offline_chat_raises():
    with pytest.raises(RuntimeError, match="no LLM configured"):
        OfflineLLM().chat([ChatMessage("user", content="x")], TOOLS)


@pytest.mark.llm
@pytest.mark.skipif(not (os.environ.get("LLM_API_KEY") or os.environ.get("LLM_API_BASE")),
                    reason="needs LLM_API_KEY or LLM_API_BASE (and LLM_MODEL) for a live tool-calling round trip")
def test_live_model_makes_a_tool_call():
    client = LiteLLMClient(os.environ.get("LLM_MODEL", "openai/gpt-4o-mini"), os.environ.get("LLM_API_BASE"),
                           os.environ.get("LLM_API_KEY"))
    reply = client.chat([ChatMessage("user", content="Click the element with id e1. Use the tool.")], TOOLS)
    assert reply.tool_calls and reply.tool_calls[0].name == "click"
    assert reply.tool_calls[0].arguments.get("id") == "e1"
