import pytest
from sitegraph.agents.navigator import Navigator
from sitegraph.browser.session import BrowserSession
from sitegraph.db import device_key
from sitegraph.embed import HashEmbedder
from sitegraph.git import GitOps
from sitegraph.graph.activity import ActivityLog
from sitegraph.normalize import domain_of
from sitegraph.policy import ExplorationPolicy
from sitegraph.schema import ensure_device_schema

pytestmark = [pytest.mark.integration, pytest.mark.browser]


def test_genesis_fork_commit_log_blame(store, settings, site_url):
    domain = domain_of(site_url)
    main = device_key(settings.uid, domain, "desktop")
    ensure_device_schema(store, main, 384)
    ActivityLog(store, main).start_run("execute", "old")
    git = GitOps(store, settings.uid, domain, settings)
    genesis = git.init_main("desktop")
    assert git.init_main("desktop") == genesis and store.exists(main + "@tag:genesis")
    br = git.fork("desktop", "explorer-1")
    assert store.exists(br.graph_key)
    assert store.ro_query(br.graph_key, "MATCH (r:Run) RETURN count(r)") == [[0]]
    writer = git.recorder(br)
    with BrowserSession() as s:
        Navigator(store, br.graph_key, domain, s, settings, HashEmbedder(), writer=writer).explore(
            site_url + "/index.html", ExplorationPolicy(max_depth=0))
    cid = git.commit(br, writer, "explorer-1", "explore index")
    assert cid and git.commit(br, writer, "explorer-1", "nothing") is None
    assert git.first_parent_chain(cid) == [cid, genesis]
    sg = store.ro_query(br.graph_key, "MATCH (s:State:Entry) RETURN s.gid")[0][0]
    blame = git.blame(sg)
    assert blame and blame[0]["commit"] == cid and blame[0]["agent"] == "explorer-1"
    assert git.log("desktop")[0]["id"] == genesis  # main untouched until merge
    assert git.branch(br.name).base_commit == genesis
    assert [c.op for _, c in git.changes_of(cid)][0] == "write_state"
    assert git.log("desktop")[0]["tags"] == ["desktop/genesis"]
