import time
import pytest
from sitegraph.embed import HashEmbedder
from sitegraph.ops import Ops
from sitegraph.policy import ExplorationPolicy
from sitegraph.profile import Profile

pytestmark = pytest.mark.integration


def test_prefs_policy_sessions_memory_sites(store, settings):
    p = Profile(store, settings.uid, HashEmbedder())
    p.set_pref("demo.format", "gif")
    assert p.get_prefs()["demo.format"] == "gif"
    assert p.default_policy().max_risk == "submit_safe"
    p.save_policy(ExplorationPolicy(name="strict", max_depth=1), is_default=True)
    assert p.default_policy().name == "strict"
    sid = p.start_session()
    p.log_request(sid, "make a gif of the menu", "done", "u:x:site:a:desktop", "rd:1")
    p.remember("user prefers short gifs for single interactions")
    p.remember("example.com shows a cookie wall")
    assert "gifs" in p.recall("does the user like gifs?", k=1)[0][0]
    p.register_site("a.test", "desktop", "u:x:site:a.test:desktop", "u:x:ops:a.test", "http://a.test/")
    assert p.sites()[0]["domain"] == "a.test"


def test_work_items_are_claimed_once_and_deduped(store, settings):
    ops = Ops(store, settings.uid, "a.test")
    ops.register_agent("nav-1", "navigator")
    ops.register_agent("nav-2", "navigator")
    g1 = ops.add_item("sc", "gk", "page", "http://a.test/", 0)
    assert ops.add_item("sc", "gk", "page", "http://a.test/", 0) == g1          # idempotent
    assert ops.add_item("sc", "gk", "page", "http://a.test/p?id=7", 1) != g1
    ops.add_item("sc", "gk", "click", "http://a.test/", 0, el_gid="el:a", pre=("el:menu",))
    a, pending_a = ops.next_item("sc", "nav-1", "gk")
    b, _ = ops.next_item("sc", "nav-2", "gk")
    c, _ = ops.next_item("sc", "nav-1", "gk")
    assert len({a["gid"], b["gid"], c["gid"]}) == 3            # no item handed out twice
    d, pending = ops.next_item("sc", "nav-2", "gk")
    assert d is None and pending is True                       # all open items are held by someone
    ops.finish_item(a["gid"], "nav-1")
    ops.finish_item(b["gid"], "nav-2")
    ops.finish_item(c["gid"], "nav-1")
    assert ops.next_item("sc", "nav-2", "gk") == (None, False)  # nothing left at all
    stats = ops.item_stats("sc")
    assert stats["done"]["count"] == 3 and stats["done"]["max_attempts"] == 1
    click = next(x for x in (a, b, c) if x["kind"] == "click")
    assert click["pre"] == ("el:menu",) and click["el_gid"] == "el:a"


def test_claim_renew_extends_lease(store, settings):
    ops = Ops(store, settings.uid, "a.test")
    ops.register_agent("nav-1", "navigator")
    ops.register_agent("nav-2", "navigator")
    assert ops.claim("st:y", "g", "nav-1", ttl_ms=200)
    ops.renew("st:y", "nav-1", ttl_ms=5000)
    time.sleep(0.3)
    assert ops.claim("st:y", "g", "nav-2", ttl_ms=200) is False   # still held thanks to the renewal
    ops.release("st:y", "nav-1")
    assert ops.claim("st:y", "g", "nav-2", ttl_ms=200) is True


def test_tasks_and_claims(store, settings):
    ops = Ops(store, settings.uid, "a.test")
    ops.register_agent("nav-1", "navigator")
    ops.register_agent("nav-2", "navigator")
    t = ops.create_task("explore", "explore a.test", device="desktop")
    ops.assign(t, "nav-1")
    ops.set_status(t, "done", {"states": 3})
    task = ops.get_task(t)
    assert task["status"] == "done" and task["result"] == {"states": 3} and task["agent"] == "nav-1"
    assert ops.claim("st:x", "g", "nav-1", ttl_ms=200) is True
    assert ops.claim("st:x", "g", "nav-2", ttl_ms=200) is False
    time.sleep(0.25)
    assert ops.claim("st:x", "g", "nav-2", ttl_ms=200) is True
