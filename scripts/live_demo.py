"""Run the Demo agent with the model from .env on a mapped site. Usage: live_demo.py <site_url> "<request>"
Prints each tool call, the script (auto-approved: the user asked for the whole run), and where the files are."""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from sitegraph.agents.demo_agent import DemoAgent
from sitegraph.agents.orchestrator import Orchestrator
from sitegraph.cli import build_runtime
from sitegraph.events import NullEvents
from sitegraph.tools.core import Toolbox

site_url, request = sys.argv[1], sys.argv[2]
OUT = Path("data/live/show") / (sys.argv[3] if len(sys.argv) > 3 else "")
OUT.mkdir(parents=True, exist_ok=True)

t0 = time.monotonic()
orig_call = Toolbox.call


def logged(self, name, args):
    res = orig_call(self, name, args)
    brief = (res.error or json.dumps(res.data, default=str, ensure_ascii=False))[:160].replace("\n", " ")
    print(f"[{time.monotonic() - t0:5.0f}s] {name} {json.dumps(self.redacted(name, args), ensure_ascii=False)[:150]}"
          f" -> {'ok' if res.ok else 'FAIL'} {brief}", flush=True)
    return res


Toolbox.call = logged

rt = build_runtime(argparse.Namespace(config="config.live.yaml", uid=None, embedder="bge"), quiet=True)
_chat = rt.llm.inner.chat


def chat_logged(messages, tools, image_png=None):
    reply = _chat(messages, tools, image_png)
    calls = [(c.name, c.arguments) for c in (reply.tool_calls or [])]
    print(f"   MODEL SAID: {(reply.content or '')[:400]!r} CALLS: {json.dumps(calls, ensure_ascii=False)[:300]}", flush=True)
    last = messages[-1]
    if last.role == "tool" and last.name == "observe" and "Mystery" in (last.content or "") and not getattr(chat_logged, "shown", False):
        chat_logged.shown = True
        print("   OBSERVE RESULT SEEN BY MODEL (Mystery page):\n" + last.content[:3000], flush=True)
    return reply


rt.llm.inner.chat = chat_logged
print("model:", getattr(rt.llm, "inner", rt.llm).__dict__.get("model", "?"), flush=True)
orch = Orchestrator(rt.settings, rt.store, rt.embedder, rt.llm, rt.blobs, NullEvents())
site = orch.register_site(site_url)
agent = DemoAgent(rt.settings, rt.store, rt.embedder, rt.llm, rt.blobs, site.key, site.domain, "desktop", orch.profile)

res = agent.run(request)
for round_ in range(8):
    if res.status != "needs_input":
        break
    print("\n===== SCRIPT SHOWN FOR APPROVAL =====\n" + (res.answer or ""), flush=True)
    (OUT / "script.md").write_text(agent.script.to_markdown() if agent.script else (res.answer or ""), encoding="utf-8")
    reply_file = OUT / "reply.txt"          # the human's answer: 'approve' or the corrections to make
    reply_file.unlink(missing_ok=True)
    print(f"WAITING for your reply in {reply_file}", flush=True)
    while not reply_file.exists():
        time.sleep(2)
    time.sleep(1)
    res = agent.reply(reply_file.read_text(encoding="utf-8").strip() or "approve")

print("\nstatus:", res.status, "|", (res.answer or "")[:600])
print("renders:", res.renders)
for fmt, ref in res.renders:
    src = rt.blobs.path(ref)
    dst = OUT / f"demo-{fmt}{''.join(src.suffixes)}"
    shutil.copyfile(src, dst)
    print("saved", dst, f"{dst.stat().st_size / 1e6:.2f} MB")
if agent.script:
    (OUT / "script.md").write_text(agent.script.to_markdown(), encoding="utf-8")
    (OUT / "script.json").write_text(agent.script.to_json(), encoding="utf-8")
print(f"total {time.monotonic() - t0:.0f}s")
