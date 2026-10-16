import json

import pytest
from sitegraph.cli import build_parser, main, make_llm
from sitegraph.config import Settings
from sitegraph.llm import LiteLLMClient, OfflineLLM, complete_json


def test_offline_llm_makes_complete_json_use_defaults():
    assert complete_json(OfflineLLM(), "sys", "user", default={"ok": True}) == {"ok": True}


def test_make_llm_picks_model_only_when_configured():
    assert isinstance(make_llm(Settings()), OfflineLLM)
    assert isinstance(make_llm(Settings(llm_api_key="k")), LiteLLMClient)
    assert isinstance(make_llm(Settings(llm_api_base="http://localhost:11434/v1")), LiteLLMClient)


def test_parser_accepts_run_and_serve_message(capsys):
    a = build_parser().parse_args(["run", "http://x.test/", "explore", "open cart", "--navigators", "3"])
    assert a.requests == ["explore", "open cart"] and a.navigators == 3 and a.embedder == "bge"
    assert main(["serve"]) == 2
    assert "not built yet" in capsys.readouterr().err


@pytest.mark.integration
@pytest.mark.browser
def test_cli_explores_then_automates_offline(store, settings, site_url, capsys, tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"data_dir: {json.dumps(str(settings.data_dir))}\nmin_target_score: 0.3\nnavigators_per_site: 2\n")
    code = main(["run", site_url + "/index.html", "explore the site", "open cart",
                 "--embedder", "hash", "--config", str(cfg), "--uid", settings.uid])
    out = capsys.readouterr().out
    assert code == 0, out
    results = [json.loads(part) for part in out.replace("}\n{", "}\x00{").split("\x00")]
    assert results[0]["kind"] == "explore" and results[0]["states"] >= 5 and results[0]["navigators"] == 2
    assert results[1]["kind"] == "automate" and results[1]["status"] == "done"


# ---------- chat command ----------
from sitegraph.agents.loop import AgentRun
from sitegraph.cli import converse
from sitegraph.llm import ChatMessage, ScriptedLLM, ToolCall


class StubAgent:
    """Stands in for ExecutionAgent: first answer needs approval, then finishes (or is told no)."""

    def __init__(self):
        self.log = []

    def ask(self, text, on_event=None):
        self.log.append(("ask", text))
        return AgentRun("needs_approval", pending=[{"id": "ap1", "tool": "click", "args": {"element_id": "e7"},
                                                    "reason": "'Checkout' is classified 'submit_mutating'"}])

    def approve(self, approval_id, on_event=None):
        self.log.append(("approve", approval_id))
        return AgentRun("done", answer="placed")

    def deny(self, approval_id, on_event=None):
        self.log.append(("deny", approval_id))
        return AgentRun("done", answer="not placed")


@pytest.mark.parametrize("reply, expected", [("y", "approve"), ("yes", "approve"), ("n", "deny"), ("", "deny"),
                                             ("maybe", "deny")])
def test_terminal_approval_prompt_defaults_to_no(reply, expected):
    agent, shown, asked = StubAgent(), [], []

    def fake_input(prompt):
        asked.append(prompt)
        return reply

    run = converse(agent, "check out", fake_input, shown.append)
    assert run.status == "done" and agent.log[-1] == (expected, "ap1")
    assert "'Checkout' is classified 'submit_mutating'" in asked[0] and "[y/N]" in asked[0]


def test_chat_without_an_llm_falls_back_to_the_fixed_pipeline(store, settings, site_url, capsys, monkeypatch, tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"data_dir: {json.dumps(str(settings.data_dir))}\n")
    lines = iter(["what should I automate?", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
    code = main(["chat", site_url + "/index.html", "--embedder", "hash", "--config", str(cfg), "--uid", settings.uid])
    out = capsys.readouterr()
    assert code == 0 and "No LLM agent" in out.err and "no LLM configured" in out.err
    assert "next_actions" in out.out                                              # the recommend pipeline answered


def test_chat_with_an_llm_runs_the_agent_and_shows_tool_calls(store, settings, site_url, capsys, monkeypatch, tmp_path):
    script = ScriptedLLM([ChatMessage("assistant", tool_calls=[ToolCall("c1", "list_known_sites", {})]),
                          ChatMessage("assistant", content="I know one site: the demo shop.")])
    monkeypatch.setattr("sitegraph.cli.make_llm", lambda s: script)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"data_dir: {json.dumps(str(settings.data_dir))}\n")
    lines = iter(["which sites do you know?", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
    code = main(["chat", site_url + "/index.html", "--embedder", "hash", "--config", str(cfg), "--uid", settings.uid])
    out = capsys.readouterr().out
    assert code == 0 and "list_known_sites" in out and "I know one site: the demo shop." in out


def test_no_agent_flag_forces_the_fixed_pipeline(store, settings, site_url, capsys, monkeypatch, tmp_path):
    monkeypatch.setattr("sitegraph.cli.make_llm", lambda s: ScriptedLLM([]))      # an LLM exists, but --no-agent wins
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"data_dir: {json.dumps(str(settings.data_dir))}\n")
    lines = iter(["recommend something", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
    code = main(["chat", site_url + "/index.html", "--embedder", "hash", "--config", str(cfg), "--uid", settings.uid,
                 "--no-agent"])
    out = capsys.readouterr()
    assert code == 0 and "--no-agent given" in out.err and "next_actions" in out.out
