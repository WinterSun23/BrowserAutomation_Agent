import pytest
from sitegraph.agents.navigator import Navigator
from sitegraph.browser.session import BrowserSession
from sitegraph.db import device_key
from sitegraph.embed import HashEmbedder
from sitegraph.git import GitOps, MergeBusy
from sitegraph.normalize import domain_of
from sitegraph.ops import Ops
from sitegraph.policy import ExplorationPolicy
from sitegraph.schema import ensure_device_schema

pytestmark = [pytest.mark.integration, pytest.mark.browser]


def _explore(store, settings, git, br, url, domain):
    w = git.recorder(br)
    with BrowserSession() as s:
        Navigator(store, br.graph_key, domain, s, settings, HashEmbedder(), writer=w).explore(
            url, ExplorationPolicy(max_depth=0))
    return w


def test_merge_two_branches_union_without_duplicates(store, settings, site_url):
    domain = domain_of(site_url)
    main = device_key(settings.uid, domain, "desktop")
    ensure_device_schema(store, main, 384)
    git = GitOps(store, settings.uid, domain, settings)
    b1, b2 = git.fork("desktop", "explorer-1"), git.fork("desktop", "explorer-2")
    w1 = _explore(store, settings, git, b1, site_url + "/index.html", domain)
    w2 = _explore(store, settings, git, b2, site_url + "/products.html", domain)
    el = store.ro_query(b1.graph_key, "MATCH (e:Element {role:'button'}) WHERE e.name_sample = 'Account' RETURN e.gid")[0][0]
    w1.bump_locator(el, "css=#from-one")
    w2.bump_locator(el, "css=#from-two")
    git.commit(b1, w1, "explorer-1", "index")
    git.commit(b2, w2, "explorer-2", "products")
    r1 = git.merge(b1)
    r2 = git.merge(b2)
    assert r1.commit and r2.commit and r1.applied > 0 and r2.applied > 0
    states = store.ro_query(main, "MATCH (s:State) RETURN count(s), count(DISTINCT s.gid)")[0]
    assert states[0] == states[1] and states[0] >= 4
    assert len(r2.conflicts) == 1 and r2.conflicts[0]["resolution"] == "theirs"
    assert store.ro_query(main, "MATCH (e:Element {gid:$g}) RETURN e.locators[0]", {"g": el}) == [["css=#from-two"]]
    assert store.ro_query(main, "MATCH (r:Run) RETURN count(r)") == [[2]]
    assert not store.exists(b1.graph_key)
    log = git.log("desktop")
    assert log[0]["msg"].startswith("merge ")
    assert len(git.ro("MATCH (:Commit {id:$c})-[:PARENT]->(p) RETURN p.id", c=r2.commit)) == 2
    assert git.ro("MATCH (x:Conflict)-[:IN_MERGE]->(:Commit {id:$c}) RETURN x.rule", c=r2.commit) == \
        [["newer-observation-wins"]]


def test_only_one_merge_into_main_at_a_time(store, settings, site_url):
    domain = domain_of(site_url)
    ensure_device_schema(store, device_key(settings.uid, domain, "desktop"), 384)
    git = GitOps(store, settings.uid, domain, settings)
    br = git.fork("desktop", "explorer-1")
    ops = Ops(store, settings.uid, domain)
    ops.register_agent("other-curator", "curator")
    assert ops.claim("merge:desktop", git.key, "other-curator", ttl_ms=60000)
    with pytest.raises(MergeBusy):
        git.merge(br, wait_s=0)
    assert store.exists(br.graph_key)                       # nothing was lost
    ops.release("merge:desktop", "other-curator")
    res = git.merge(br)                                     # an empty branch still merges (a 2-parent commit)
    assert res.commit and res.applied == 0 and not store.exists(br.graph_key)
    ran = []
    br2 = git.fork("desktop", "explorer-2")
    git.merge(br2, recompute=lambda: ran.append(1), keep_branch=True)
    assert ran == [1] and store.exists(br2.graph_key)
