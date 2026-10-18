from pathlib import Path

from sitegraph.cli import build_parser
from sitegraph.eval.report import load_runs, summarize, write_reports
from sitegraph.eval.runner import TaskRun
from sitegraph.eval.suite import load_suite

PUBLIC = Path(__file__).parent.parent / "eval" / "suites" / "public.yaml"


def _run(task, outcome, variant="v", rep=0, seconds=10.0, tokens=100):
    return TaskRun(task, variant, rep, outcome, failed=[] if outcome == "pass" else ["x"], seconds=seconds,
                   tokens=tokens, error="boom" if outcome == "error" else None)


def test_summary_counts_errors_apart_and_finds_flaky_tasks():
    runs = [_run("a", "pass", rep=0, seconds=10), _run("a", "fail", rep=1, seconds=20), _run("a", "pass", rep=2, seconds=30),
            _run("b", "pass", rep=0), _run("b", "error", rep=1), _run("b", "pass", rep=2),
            _run("c", "skip", rep=0)]
    v = summarize(runs)["v"]
    assert v["errors"] == 1 and v["skipped"] == 1
    assert v["pass_rate"] == 4 / 5                    # errors and skips are not in the denominator
    assert v["flaky"] == ["a"]                        # b's error is not flakiness
    assert v["median_seconds"] == 10.0 and v["median_tokens"] == 100


def test_reports_list_regressions_against_a_baseline(tmp_path):
    baseline = [_run("cart", "pass"), _run("explore", "pass")]
    runs = [_run("cart", "fail"), _run("explore", "pass")]
    md, html = write_reports(runs, tmp_path, baseline)
    text = md.read_text(encoding="utf-8")
    assert "## Regressions" in text
    regressions = text.split("## Regressions", 1)[1].split("\n## ", 1)[0]
    assert "cart" in regressions and "explore" not in regressions
    page = html.read_text(encoding="utf-8")
    assert "<table" in page and "cart" in page and "Regressions" in page
    assert "http://" not in page.replace("http://www.w3.org", "") and "https://" not in page  # self-contained
    assert md.parent == tmp_path and html.name == "report.html"


def test_load_runs_reads_results_jsonl(tmp_path):
    import dataclasses
    import json
    runs = [_run("a", "pass"), _run("b", "error")]
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(dataclasses.asdict(r)) + "\n" for r in runs),
                                            encoding="utf-8")
    assert load_runs(tmp_path) == runs


def test_public_suite_loads():
    suite = load_suite(PUBLIC)
    sites = {t.site.split("/")[2] for t in suite.tasks}
    assert len(suite.tasks) >= 12 and len(sites) >= 4
    assert all(t.checks for t in suite.tasks)
    assert any(t.requires for t in suite.tasks)       # tasks waiting for later features are marked
    cold = [t for t in suite.tasks if t.id.endswith("-cold")]
    assert len(cold) >= 5 and all(not t.setup and t.max_tokens for t in cold)
    assert all({"result": {"path": "goal_explore.found", "op": "==", "value": True}} in t.checks for t in cold)
    assert all({"result": {"path": "states", "op": "<=", "value": 40}} in t.checks for t in cold)


def test_modes_suite_declares_brute_and_auto():
    suite = load_suite(PUBLIC.parent / "modes.yaml")
    assert suite.variants == {"brute": {"exploration_mode": "brute"}, "auto": {"exploration_mode": "auto"}}
    assert len(suite.tasks) >= 4 and all(t.request == "explore the site" for t in suite.tasks)


def test_exploration_table_compares_variants_per_explore_task(tmp_path):
    def ex(task, variant, rep, states, seconds, tokens, kind="explore", outcome="pass"):
        return TaskRun(task, variant, rep, outcome, seconds=seconds, tokens=tokens, states=states, actions=states * 2,
                       kind=kind)
    runs = [ex("books-explore", "brute", 0, 30, 100.0, 0), ex("books-explore", "brute", 1, 34, 120.0, 0),
            ex("books-explore", "auto", 0, 12, 60.0, 4000), ex("books-explore", "auto", 1, 99, 1.0, 1, outcome="error"),
            ex("cart", "brute", 0, 0, 5.0, 0, kind="automate")]
    md, html = write_reports(runs, tmp_path)
    text = md.read_text(encoding="utf-8")
    table = text.split("## Exploration", 1)[1].split("\n## ", 1)[0]
    assert "| books-explore | 32 / 64 / 110.0 / 0 | 12 / 24 / 60.0 / 4000 |" in table   # medians, errors left out
    assert "cart" not in table and "brute" in table and "auto" in table
    assert "Exploration" in html.read_text(encoding="utf-8") and "110.0" in html.read_text(encoding="utf-8")
    md2, _ = write_reports([ex("cart", "v", 0, 0, 5.0, 0, kind="automate")], tmp_path / "b")
    assert "## Exploration" not in md2.read_text(encoding="utf-8")


def test_eval_and_mode_flags_parse():
    a = build_parser().parse_args(["eval", "s.yaml", "--variant", "brute:exploration_mode=brute,max_replans=1",
                                   "--variant", "auto:exploration_mode=auto", "--repeat", "3", "--out", "o",
                                   "--keep", "--site-url", "http://127.0.0.1:1", "--compare", "old", "-v"])
    assert a.cmd == "eval" and a.repeat == 3 and a.keep and a.compare == "old" and a.verbose == 1
    assert a.variant == ["brute:exploration_mode=brute,max_replans=1", "auto:exploration_mode=auto"]
    r = build_parser().parse_args(["run", "http://x.test/", "explore", "--mode", "brute"])
    assert r.mode == "brute"
    assert build_parser().parse_args(["run", "http://x.test/", "explore"]).mode is None


def test_parse_variants():
    from sitegraph.cli import parse_variants
    assert parse_variants([]) == {"default": {}}
    assert parse_variants(["brute:exploration_mode=brute,max_replans=1", "plain"]) == {
        "brute": {"exploration_mode": "brute", "max_replans": "1"}, "plain": {}}
