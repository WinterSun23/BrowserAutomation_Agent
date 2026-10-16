import asyncio
import json
import sys

import pytest
from mcp import Client
from mcp import types as mt
from mcp.client.stdio import StdioServerParameters

from sitegraph.config import Settings
from sitegraph.mcp_server import NOT_FOR_MCP, build_server
from sitegraph.tools.core import ToolContext, ToolResult, ToolSpec, Toolbox


def _box():
    box = Toolbox(ToolContext(settings=Settings()))
    box.register(ToolSpec("echo", "Echo a message", {"type": "object", "properties": {"msg": {"type": "string"}},
                                                     "required": ["msg"]}, lambda c, a: {"said": a["msg"]}, "read"))
    box.register(ToolSpec("boom", "Always fails", {"type": "object", "properties": {}},
                          lambda c, a: (_ for _ in ()).throw(RuntimeError("element gone")), "act"))
    box.register(ToolSpec("wipe", "Delete everything", {"type": "object", "properties": {}},
                          lambda c, a: {"wiped": True}, "risky"))
    box.register(ToolSpec("ask_user", "Ask", {"type": "object", "properties": {}},
                          lambda c, a: ToolResult(True, pause="which?"), "read"))
    box.register(ToolSpec("peek", "Return an image", {"type": "object", "properties": {}},
                          lambda c, a: ToolResult(True, data={"ok": 1}, image_png=b"\x89PNG-bytes", image_label="x"), "read"))
    return box


def run(coro):
    return asyncio.run(coro)


def text_of(result: mt.CallToolResult) -> str:
    return "\n".join(c.text for c in result.content if isinstance(c, mt.TextContent))


def test_tools_list_matches_the_toolbox_without_the_chat_only_tools():
    box = _box()

    async def go():
        async with Client(build_server(box)) as client:
            return (await client.list_tools()).tools

    tools = run(go())
    names = {t.name for t in tools}
    assert names == {s["function"]["name"] for s in box.specs()} - NOT_FOR_MCP
    echo = next(t for t in tools if t.name == "echo")
    assert echo.input_schema["required"] == ["msg"] and echo.description == "Echo a message"


def test_call_round_trip_errors_and_images():
    async def go():
        async with Client(build_server(_box())) as client:
            ok = await client.call_tool("echo", {"msg": "hi"})
            bad_args = await client.call_tool("echo", {})
            boom = await client.call_tool("boom", {})
            unknown = await client.call_tool("ask_user", {})
            image = await client.call_tool("peek", {})
            return ok, bad_args, boom, unknown, image

    ok, bad_args, boom, unknown, image = run(go())
    assert not ok.is_error and '"said": "hi"' in text_of(ok)
    assert bad_args.is_error and "msg" in text_of(bad_args)
    assert boom.is_error and "element gone" in text_of(boom)
    assert unknown.is_error and "unknown tool" in text_of(unknown)                    # hidden from MCP
    assert any(isinstance(c, mt.ImageContent) and c.mime_type == "image/png" for c in image.content)


def test_risky_call_without_elicitation_is_refused_with_an_explanation():
    box = _box()

    async def go():
        async with Client(build_server(box)) as client:                                # a client that cannot ask the user
            return await client.call_tool("wipe", {})

    res = run(go())
    assert res.is_error and "cannot ask the user" in text_of(res) and "wiped" not in text_of(res)
    assert box.ctx.approvals.pending()                                                 # still waiting for a human


def _client_that_answers(box, allow: bool, asked: list):
    async def on_elicit(context, params):
        asked.append(params.message)
        return mt.ElicitResult(action="accept", content={"allow": allow})

    return Client(build_server(box), elicitation_callback=on_elicit)


def test_risky_call_runs_once_when_the_human_approves_through_elicitation():
    box, asked = _box(), []

    async def go():
        async with _client_that_answers(box, True, asked) as client:
            first = await client.call_tool("wipe", {})
            second = await client.call_tool("wipe", {})
            return first, second

    first, second = run(go())
    assert not first.is_error and '"wiped": true' in text_of(first) and "Delete everything" in asked[0]
    assert len(asked) == 2 and not second.is_error                                       # asked again: approval was one-time


def test_risky_call_stays_refused_when_the_human_declines():
    box, asked = _box(), []

    async def go():
        async with _client_that_answers(box, False, asked) as client:
            return await client.call_tool("wipe", {})

    res = run(go())
    assert res.is_error and "declined" in text_of(res) and "wiped" not in text_of(res)


@pytest.mark.integration
@pytest.mark.browser
def test_stdio_server_serves_the_real_toolbox(store, settings, site_url, tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"data_dir: {json.dumps(str(settings.data_dir))}\n")
    params = StdioServerParameters(command=sys.executable, args=[
        "-m", "sitegraph", "mcp", site_url + "/index.html", "--embedder", "hash", "--config", str(cfg),
        "--uid", settings.uid])

    async def go():
        async with Client(params) as client:
            tools = {t.name for t in (await client.list_tools()).tools}
            opened = await client.call_tool("open_url", {"url": site_url + "/index.html"})
            seen = await client.call_tool("observe", {})
            external = await client.call_tool("open_url", {"url": "http://example.org/"})
            return tools, opened, seen, external

    tools, opened, seen, external = run(go())
    assert {"observe", "click", "find_screens", "run_goal", "explore"} <= tools and "ask_user" not in tools
    assert not opened.is_error and not seen.is_error and "Products" in text_of(seen)
    assert external.is_error and "cannot ask the user" in text_of(external)
