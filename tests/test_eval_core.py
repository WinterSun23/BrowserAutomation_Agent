import json
from pathlib import Path

import pytest

from sitegraph.embed import HashEmbedder
from sitegraph.eval.checks import CheckContext, check
from sitegraph.eval.runner import run_suite
from sitegraph.eval.suite import EvalTask, Suite, load_suite
from sitegraph.llm import FakeLLM, TracingLLM

OFFLINE = Path(__file__).parent.parent / "eval" / "suites" / "offline.yaml"


def _ctx(**kw):
    base = dict(result={"status": "done", "states": 12, "kind": "automate", "nested": {"n": 3}},
                final_url="http://shop.test/cart.html", page_text="Your Cart\nTotal: 51.77", answer="It costs £51.77",
                seconds=12.0, tokens=900)
    return CheckContext(**{**base, **kw})


def test_load_suite_reads_tasks_and_rejects_unknown_checks(tmp_path):
    good = tmp_path / "good.yaml"
    good.write_text("name: s\ndefaults: {min_target_score: 0.3}\ntasks:\n"
                    "  - id: one\n    site: '{site}/index.html'\n    request: open cart\n"
                    "    setup: [explore the site]\n    checks:\n      - url_matches: cart\n"
                    "      - result: status == done\n", encoding="utf-8")
    suite = load_suite(good, site_url="http://127.0.0.1:8000")
    assert suite.name == "s" and suite.defaults == {"min_target_score": 0.3}
    t = suite.tasks[0]
    assert t.site == "http://127.0.0.1:8000/index.html" and t.setup == ["explore the site"] and t.max_seconds == 600
    assert t.checks[1] == {"result": {"path": "status", "op": "==", "value": "done"}}
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: s\ntasks:\n  - id: broken\n    site: http://x.test/\n    request: r\n"
                   "    checks:\n      - smells_nice: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="broken"):
        load_suite(bad)
    no_checks = tmp_path / "none.yaml"
    no_checks.write_text("name: s\ntasks:\n  - id: empty\n    site: http://x.test/\n    request: r\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_suite(no_checks)


def test_each_check_kind():
    c = _ctx()
    assert check({"url_matches": r"cart\.html$"}, c).ok
    assert not check({"url_matches": "checkout"}, c).ok
    assert check({"page_contains": "your cart"}, c).ok                    # case-insensitive
    assert not check({"page_contains": "empty"}, c).ok
    assert check({"answer_contains_any": ["99.99", "51.77"]}, c).ok
    assert not check({"answer_contains_any": ["99.99"]}, c).ok
    assert check({"result": {"path": "status", "op": "==", "value": "done"}}, c).ok
    assert check({"result": "status != failed"}, c).ok
    assert check({"result": "states >= 5"}, c).ok and not check({"result": "states < 5"}, c).ok
    assert check({"result": "states <= 12"}, c).ok and check({"result": "states > 11"}, c).ok
    assert check({"result": {"path": "kind", "op": "in", "value": ["automate", "ask"]}}, c).ok
    assert check({"result": "nested.n == 3"}, c).ok
    r = check({"result": "missing.path == 1"}, c)
    assert not r.ok and "missing.path" in r.detail
    assert check({"max_seconds": 30}, c).ok and not check({"max_seconds": 5}, c).ok
    assert check({"max_tokens": 1000}, c).ok and not check({"max_tokens": 500}, c).ok


def test_result_checks_read_nested_paths_and_fall_back_to_run_metrics():
    c = _ctx(result={"kind": "automate", "status": "done", "goal_explore": {"found": True, "how": "inputs"}},
             metrics={"states": 14, "tokens": 900})
    assert check({"result": "goal_explore.found == true"}, c).ok
    assert check({"result": "goal_explore.how == inputs"}, c).ok
    assert not check({"result": "goal_explore.how == rounds"}, c).ok
    assert check({"result": "states <= 40"}, c).ok and not check({"result": "states <= 10"}, c).ok   # from metrics
    no_search = _ctx(result={"kind": "automate", "goal_explore": None})          # the map already had the target
    r = check({"result": "goal_explore.found == true"}, no_search)
    assert not r.ok and r.detail == "result has no goal_explore.found"
    assert not check({"result": "goal_explore.found == true"}, _ctx(result={"goal_explore": {"found": False}})).ok
    explore = _ctx(result={"kind": "explore", "states": 30}, metrics={"states": 2})
    assert check({"result": "states >= 30"}, explore).ok                          # the result wins over metrics


def test_suite_variants_are_read(tmp_path):
    f = tmp_path / "m.yaml"
    f.write_text("name: m\nvariants:\n  brute: {exploration_mode: brute}\n  plain:\ntasks:\n"
                 "  - id: e\n    site: http://x.test/\n    request: explore\n    checks: [{result: states >= 1}]\n",
                 encoding="utf-8")
    assert load_suite(f).variants == {"brute": {"exploration_mode": "brute"}, "plain": {}}
    f.write_text("name: m\nvariants: [brute]\ntasks: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="variants"):
        load_suite(f)


def test_tracing_llm_counts_calls_and_tokens(tmp_path):
    class Counting(FakeLLM):
        def __init__(self):
            super().__init__(lambda s, u: "{}")
            self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

        def complete(self, system, user, image_png=None):
            self.usage["prompt_tokens"] += 10
            self.usage["completion_tokens"] += 5
            return super().complete(system, user, image_png)

    t = TracingLLM(Counting(), tmp_path)
    t.complete("s", "u")
    t.complete("s", "u")
    assert t.counters() == {"calls": 2, "tokens": 30}
    t.reset_counters()
    t.complete("s", "u")
    assert t.counters() == {"calls": 1, "tokens": 15}
    plain = TracingLLM(FakeLLM(), tmp_path)                     # no usage on the inner model: tokens stay 0
    plain.complete("s", "u")
    assert plain.counters() == {"calls": 1, "tokens": 0}


def _offline_suite(site_url):
    return load_suite(OFFLINE, site_url=site_url)


@pytest.mark.integration
@pytest.mark.browser
def test_offline_suite_runs_and_writes_results(store, settings, site_url, tmp_path):
    suite = _offline_suite(site_url)
    assert [t.id for t in suite.tasks] == ["explore", "cart", "cart-cold"]
    runs = run_suite(suite, {"default": {}}, 1, tmp_path / "out", settings, llm_factory=lambda s: FakeLLM(),
                     embedder=HashEmbedder(settings.embed_dim))
    lines = [json.loads(x) for x in (tmp_path / "out" / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 3 and len(runs) == 3
    assert [r["outcome"] for r in lines] == ["pass", "pass", "pass"], lines
    assert [r["kind"] for r in lines] == ["explore", "automate", "automate"]
    assert all(r["tokens"] >= 0 for r in lines)
    explore = next(r for r in lines if r["task_id"] == "explore")
    assert explore["states"] >= 5 and explore["variant"] == "default" and explore["rep"] == 0


@pytest.mark.integration
def test_a_site_that_is_down_is_an_error_not_a_failure(store, settings, site_url, tmp_path):
    suite = Suite("down", [EvalTask("down", "http://127.0.0.1:9/", "explore the site", [], [{"result": "states >= 1"}]),
                           EvalTask("up", site_url + "/index.html", "remember that I like widgets", [],
                                    [{"result": "kind == remember"}])], {})
    runs = run_suite(suite, {"default": {}}, 1, tmp_path / "out", settings, llm_factory=lambda s: FakeLLM(),
                     embedder=HashEmbedder(settings.embed_dim))
    assert [r.outcome for r in runs] == ["error", "pass"]
    assert runs[0].error
    assert len((tmp_path / "out" / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 2


@pytest.mark.integration
def test_each_run_uses_its_own_namespace_and_cleans_up(store, settings, site_url, tmp_path):
    before = set(store.list_graphs("u:eval-"))
    suite = Suite("ns", [EvalTask("mem", site_url + "/index.html", "remember that I like widgets", [],
                                  [{"result": "kind == remember"}])], {})
    runs = run_suite(suite, {"a": {}, "b": {"max_replans": 1}}, 2, tmp_path / "out", settings,
                     llm_factory=lambda s: FakeLLM(), embedder=HashEmbedder(settings.embed_dim))
    assert len(runs) == 4 and all(r.outcome == "pass" for r in runs)
    assert {(r.variant, r.rep) for r in runs} == {("a", 0), ("a", 1), ("b", 0), ("b", 1)}
    assert set(store.list_graphs("u:eval-")) - before == set()


def test_a_task_that_needs_an_unsupported_feature_is_skipped(tmp_path, settings):
    suite = Suite("skip", [EvalTask("sel", "http://127.0.0.1:9/", "pick sort order", [], [{"result": "status == done"}],
                                    requires=["no-such-feature"])], {})
    runs = run_suite(suite, {"default": {}}, 1, tmp_path / "out", settings, llm_factory=lambda s: FakeLLM(),
                     embedder=HashEmbedder(settings.embed_dim))
    assert runs[0].outcome == "skip" and "no-such-feature" in (runs[0].error or "")
