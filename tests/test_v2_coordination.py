import json
import pytest
from sitegraph.agents.orchestrator import Orchestrator
from sitegraph.blobs import BlobStore
from sitegraph.embed import HashEmbedder
from sitegraph.events import NullEvents
from sitegraph.git import GitOps
from sitegraph.llm import FakeLLM
from sitegraph.ops import Ops
from tests.helpers import site_responder

pytestmark = [pytest.mark.integration, pytest.mark.browser]


def _llm():
    def r(system, user):
        if "Classify" in user:
            request = user.split("\n", 1)[0].lower()      # the rest of the prompt lists every kind, "explore" included
            return json.dumps({"kind": "explore"} if "explore" in request else
                              {"kind": "automate", "goal": "open cart", "params": {}})
        return site_responder(system, user)
    return FakeLLM(r)


def test_git_explore_parallel_exec_commits_handoff_notes(store, settings, site_url):
    settings.min_target_score = 0.3
    settings.dynamic_spawn = False        # this test is about merging two branches, so both must run
    o = Orchestrator(settings, store, HashEmbedder(), _llm(), BlobStore(settings.blob_dir), NullEvents())
    site = o.register_site(site_url + "/index.html")
    ops = Ops(store, settings.uid, site.domain)
    t1 = ops.create_task("chat", "explore the site")
    r1 = o.run_task(t1, "explore the site", site)
    assert r1["kind"] == "explore" and r1["merge_commit"], r1
    git = GitOps(store, settings.uid, site.domain, settings)
    assert git.log("desktop")[0]["msg"].startswith("merge ")
    t2 = ops.create_task("chat", "explore more", depends_on=[t1])
    assert [t["id"] for t in ops.runnable()] == [t2]
    r2 = o.explore_parallel(site, o.profile.policy_for(site.domain, site.device), t2, n=2)
    assert r2["merges"] == 2, (r2["per_agent"], r2["failed_agents"])
    pages = store.ro_query(site.key, "MATCH (p:Page) RETURN count(p), count(DISTINCT p.gid)")[0]
    assert pages[0] == pages[1]
    t3 = ops.create_task("chat", "take me to the cart")
    r3 = o.run_task(t3, "take me to the cart", site)
    assert r3.get("status") == "done" and r3["exec_commit"], r3
    br = git.fork("desktop", "explorer-a")
    br2 = git.handoff(br, "explorer-a", "explorer-b", task_id=t3, reason="budget exhausted", context={"next": "cart"})
    assert br2.owner == "explorer-b"
    assert ops.q("MATCH (h:Handoff)-[:TO_AGENT]->(:Agent {id:'explorer-b'}) RETURN count(h)") == [[1]]
    ops.add_note("login wall at /account; skipped", "explorer-a", HashEmbedder())
    assert "login wall" in ops.notes("where is the login wall?", HashEmbedder(), k=1)[0][0]
    roles = {r[0] for r in ops.q("MATCH (a:Agent) RETURN a.role")}
    assert {"navigator", "curator", "planner", "executor", "explorer"} <= roles
