import json
import re
import pytest
from sitegraph.agents.orchestrator import Orchestrator
from sitegraph.blobs import BlobStore
from sitegraph.demo.director import DemoDirector
from sitegraph.demo.storyboard import load_storyboard
from sitegraph.embed import HashEmbedder
from sitegraph.events import NullEvents
from sitegraph.git import GitOps
from sitegraph.llm import FakeLLM
from sitegraph.ops import Ops
from sitegraph.tools import ToolContext, build_toolbox
from tests.helpers import site_responder

pytestmark = [pytest.mark.integration, pytest.mark.browser]


def test_blame_in_explanation_and_pinned_rerender(store, settings, site_url):
    settings.min_target_score = 0.3

    def r(system, user):
        if "Classify" in user:
            request = user.split("\n", 1)[0].lower()      # the prompt lists every kind, "explore" included
            return json.dumps({"kind": "explore"} if "explore" in request else
                              {"kind": "automate", "goal": "open cart", "params": {}})
        if user.startswith("Request:"):
            return json.dumps({"formats": ["screenshot"], "scope_kind": "goal", "target": "open cart"})
        return site_responder(system, user)

    o = Orchestrator(settings, store, HashEmbedder(), FakeLLM(r), BlobStore(settings.blob_dir), NullEvents())
    site = o.register_site(site_url + "/index.html")
    ops = Ops(store, settings.uid, site.domain)
    o.run_task(ops.create_task("chat", "explore"), "explore the site", site)
    res = o.run_task(ops.create_task("chat", "cart"), "take me to the cart", site)
    assert re.search(r"discovered by explorer-desktop-[01] in c:", res["explanation"]), res
    git = GitOps(store, settings.uid, site.domain, settings)
    d = DemoDirector(store, site.key, site.domain, settings, HashEmbedder(), FakeLLM(r), o.blobs, git=git)
    first = d.run("screenshot of the cart")
    sb = load_storyboard(store, site.key, first.storyboard_gid)
    assert sb.map_commit == git.head("desktop")
    assert any(c["msg"].startswith("demo capture") for c in git.log("desktop")[:3])
    again = d.rerender(sb.gid, pinned=True)
    assert again.renders
    assert store.ro_query(site.key, "MATCH (s:Storyboard {gid:$g}) RETURN s.version", {"g": sb.gid}) == [[2]]
    assert not [g for g in store.list_graphs(site.key) if "@at:" in g]          # the throwaway checkout is gone
    # the git tools see the same history, and reverting needs the user's approval
    ctx = ToolContext(settings=settings, store=store, key=site.key, domain=site.domain, embedder=HashEmbedder(),
                      llm=FakeLLM(r), blobs=o.blobs, device="desktop")
    box = build_toolbox(ctx)
    log = box.call("git_log", {"limit": 5})
    assert log.ok and log.data[0]["id"] == git.head("desktop")
    assert box.call("blame", {"gid": sb.workflow_gid or log.data[0]["id"]}).ok
    blocked = box.call("git_revert", {"commit": log.data[0]["id"]})
    assert not blocked.ok and blocked.needs_approval and git.head("desktop") == log.data[0]["id"]
    assert box.call("git_tag", {"name": "before-test"}).ok
    ctx.close()
