import threading
import time

from sitegraph.agents.spawner import run_scaled


def _run(agents, run_one, backlog, **kw):
    kw.setdefault("min_backlog", 3)
    kw.setdefault("after_s", 0.15)
    kw.setdefault("poll_s", 0.02)
    return run_scaled(agents, run_one, backlog, **kw)


def test_extra_agents_start_only_after_the_backlog_is_sustained():
    started, lock = {}, threading.Lock()

    def run_one(agent):
        with lock:
            started[agent] = time.monotonic()
        time.sleep(0.8)
        return agent

    t0 = time.monotonic()
    outs, errors, spawned = _run(["a", "b", "c"], run_one, lambda: 10)
    assert spawned == ["a", "b", "c"] and not errors
    assert [o[0] for o in outs] == ["a", "b", "c"] and [o[1] for o in outs] == ["a", "b", "c"]
    assert started["a"] - t0 < 0.1                         # the first one starts at once
    assert started["b"] - started["a"] >= 0.14             # each further one waits for the backlog to persist
    assert started["c"] - started["b"] >= 0.14             # and the timer restarts after every spawn


def test_no_extra_agent_when_the_backlog_is_small():
    outs, errors, spawned = _run(["a", "b", "c"], lambda a: (time.sleep(0.5), a)[1], lambda: 2)
    assert spawned == ["a"] and [o[0] for o in outs] == ["a"]


def test_a_short_spike_does_not_spawn():
    t0 = time.monotonic()
    outs, errors, spawned = _run(["a", "b"], lambda a: (time.sleep(0.6), a)[1],
                                 lambda: 10 if time.monotonic() - t0 < 0.1 else 0, after_s=0.3)
    assert spawned == ["a"]


def test_never_more_than_the_allowed_agents():
    outs, errors, spawned = _run(["a", "b"], lambda a: (time.sleep(0.6), a)[1], lambda: 99)
    assert spawned == ["a", "b"]


def test_nothing_more_is_started_once_the_running_agents_are_done():
    outs, errors, spawned = _run(["a", "b", "c"], lambda a: a, lambda: 0, after_s=0.05)
    assert spawned == ["a"] and outs == [("a", "a")]


def test_a_failing_agent_does_not_lose_the_others():
    def run_one(agent):
        time.sleep(0.4)
        if agent == "b":
            raise RuntimeError("boom")
        return agent

    outs, errors, spawned = _run(["a", "b"], run_one, lambda: 10)
    assert spawned == ["a", "b"] and [o[0] for o in outs] == ["a"] and len(errors) == 1
    assert "boom" in str(errors[0])


def test_a_broken_backlog_query_counts_as_no_backlog():
    def backlog():
        raise RuntimeError("db down")

    outs, errors, spawned = _run(["a", "b"], lambda a: (time.sleep(0.4), a)[1], backlog)
    assert spawned == ["a"] and not errors


def test_backlog_counts_only_open_unclaimed_items(store, settings):
    from sitegraph.ops import Ops

    ops = Ops(store, settings.uid, "backlog.test")
    ops.register_agent("a1", "navigator")
    for i in range(5):
        ops.add_item("sc", "g", "page", f"http://backlog.test/p{i}", 1)
    assert ops.backlog("sc") == 5
    item, _ = ops.next_item("sc", "a1", "g")                    # claimed: still open, but someone is on it
    assert ops.backlog("sc") == 4
    ops.finish_item(item["gid"], "a1")
    assert ops.backlog("sc") == 4
    assert ops.backlog("other-scope") == 0
