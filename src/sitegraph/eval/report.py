"""Evaluation reports: summary per variant, a task x variant table, flaky tasks and regressions vs a baseline."""
from __future__ import annotations

import dataclasses
import html
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .runner import TaskRun

_FIELDS = {f.name for f in dataclasses.fields(TaskRun)}


def load_runs(run_dir) -> list[TaskRun]:
    """The TaskRuns of an earlier run (its results.jsonl; a path to the file itself works too)."""
    path = Path(run_dir)
    if path.is_dir():
        path = path / "results.jsonl"
    runs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            runs.append(TaskRun(**{k: v for k, v in d.items() if k in _FIELDS}))
    return runs


def _median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _by(runs: list[TaskRun], *attrs) -> dict:
    out: dict = defaultdict(list)
    for r in runs:
        out[tuple(getattr(r, a) for a in attrs) if len(attrs) > 1 else getattr(r, attrs[0])].append(r)
    return out


def summarize(runs: list[TaskRun]) -> dict:
    """Per variant: pass_rate (errors and skips are not in the denominator), median seconds and tokens of the
    measured runs, errors, skipped, runs, and flaky tasks (both pass and fail across repeats)."""
    out = {}
    for variant, rs in _by(runs, "variant").items():
        measured = [r for r in rs if r.outcome in ("pass", "fail")]
        flaky = sorted(t for t, tr in _by(measured, "task_id").items() if {r.outcome for r in tr} == {"pass", "fail"})
        out[variant] = {
            "runs": len(rs),
            "pass_rate": (sum(r.outcome == "pass" for r in measured) / len(measured)) if measured else None,
            "median_seconds": _median([r.seconds for r in measured]),
            "median_tokens": _median([r.tokens for r in measured]),
            "errors": sum(r.outcome == "error" for r in rs),
            "skipped": sum(r.outcome == "skip" for r in rs),
            "flaky": flaky,
        }
    return out


def _cell(rs: list[TaskRun]) -> str:
    """'2/3 pass', 'error', 'skip' for one task in one variant."""
    measured = [r for r in rs if r.outcome in ("pass", "fail")]
    if not measured:
        return rs[0].outcome if rs else "-"
    text = f"{sum(r.outcome == 'pass' for r in measured)}/{len(measured)} pass"
    errors = sum(r.outcome == "error" for r in rs)
    return text + (f" (+{errors} error)" if errors else "")


def _status(rs: list[TaskRun]) -> str | None:
    measured = [r.outcome for r in rs if r.outcome in ("pass", "fail")]
    if not measured:
        return None
    return "pass" if all(o == "pass" for o in measured) else "fail"


def regressions(runs: list[TaskRun], baseline: list[TaskRun]) -> list[tuple[str, str]]:
    """(task_id, variant) that passed every measured repeat in the baseline and failed at least once now."""
    old, new = _by(baseline, "task_id", "variant"), _by(runs, "task_id", "variant")
    return sorted(k for k, rs in new.items() if _status(old.get(k, [])) == "pass" and _status(rs) == "fail")


def _fmt(v, kind: str = "") -> str:
    if v is None:
        return "-"
    if kind == "rate":
        return f"{v:.0%}"
    return f"{v:.1f}" if isinstance(v, float) and kind == "s" else f"{v:.0f}" if isinstance(v, float) else str(v)


def _delta(new, old, kind: str = "") -> str:
    if new is None or old is None:
        return ""
    d = new - old
    if kind == "rate":
        return f" ({d * 100:+.0f} pts)"
    return f" ({d:+.1f})" if kind == "s" else f" ({d:+.0f})"


def exploration_rows(runs: list[TaskRun]) -> tuple[list[str], list[list[str]]]:
    """Per explore task, the median states / actions / seconds / tokens of each variant side by side (measured runs
    only), so exploration modes can be compared site by site."""
    variants = list(dict.fromkeys(r.variant for r in runs))
    explore = list(dict.fromkeys(r.task_id for r in runs if r.kind == "explore"))
    cells = _by([r for r in runs if r.outcome in ("pass", "fail")], "task_id", "variant")
    rows = []
    for t in explore:
        row = [t]
        for v in variants:
            rs = cells.get((t, v), [])
            row.append(" / ".join(_fmt(_median([getattr(r, a) for r in rs]), "s" if a == "seconds" else "")
                                  for a in ("states", "actions", "seconds", "tokens")) if rs else "-")
        rows.append(row)
    return ["task (states / actions / s / tokens)"] + variants, rows


def _tables(runs: list[TaskRun], baseline: list[TaskRun] | None):
    """Rows shared by the Markdown and the HTML report."""
    summary, base = summarize(runs), summarize(baseline) if baseline else {}
    head = ["variant", "pass rate", "median s", "median tokens", "errors", "skipped", "runs", "flaky"]
    srows = []
    for v, s in summary.items():
        b = base.get(v, {})
        srows.append([v, _fmt(s["pass_rate"], "rate") + _delta(s["pass_rate"], b.get("pass_rate"), "rate"),
                      _fmt(s["median_seconds"], "s") + _delta(s["median_seconds"], b.get("median_seconds"), "s"),
                      _fmt(s["median_tokens"]) + _delta(s["median_tokens"], b.get("median_tokens")),
                      str(s["errors"]), str(s["skipped"]), str(s["runs"]), ", ".join(s["flaky"]) or "-"])
    variants = list(summary)
    cells = _by(runs, "task_id", "variant")
    tasks = list(dict.fromkeys(r.task_id for r in runs))
    trows = [[t] + [_cell(cells.get((t, v), [])) for v in variants] for t in tasks]
    failures = [[r.task_id, r.variant, str(r.rep), r.outcome, "; ".join(r.failed) or (r.error or "")]
                for r in runs if r.outcome in ("fail", "error")]
    regs = [[t, v] for t, v in regressions(runs, baseline)] if baseline is not None else None
    return (head, srows), (["task"] + variants, trows), failures, regs, exploration_rows(runs)


def _md_table(head: list[str], rows: list[list[str]]) -> str:
    esc = lambda c: str(c).replace("|", "\\|").replace("\n", " ")  # noqa: E731
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(esc(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _html_table(head: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{html.escape(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


_CSS = """body{font:14px/1.45 system-ui,sans-serif;margin:24px;color:#1b1b1b;background:#fff}
table{border-collapse:collapse;margin:8px 0 20px}th,td{border:1px solid #ccc;padding:4px 8px;text-align:left;
vertical-align:top}th{background:#f2f2f2}h1{font-size:20px}h2{font-size:16px;margin-top:24px}
@media (prefers-color-scheme: dark){body{background:#161616;color:#e6e6e6}th{background:#262626}th,td{border-color:#444}}"""


def write_reports(runs: list[TaskRun], out_dir, baseline: list[TaskRun] | None = None) -> tuple[Path, Path]:
    """report.md and report.html (self-contained, no external assets) in `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (shead, srows), (thead, trows), failures, regs, (xhead, xrows) = _tables(runs, baseline)
    fhead = ["task", "variant", "rep", "outcome", "why"]
    md = ["# Evaluation report", "", f"{len(runs)} task runs." +
          (" Deltas in brackets are against the baseline run." if baseline is not None else ""), "",
          "## Summary", "", _md_table(shead, srows), "", "## Tasks", "", _md_table(thead, trows), ""]
    if xrows:
        md += ["## Exploration", "", "Medians per variant (measured runs).", "", _md_table(xhead, xrows), ""]
    if regs is not None:
        md += ["## Regressions", "", "Passed in the baseline, failed now.", "",
               _md_table(["task", "variant"], regs) if regs else "None.", ""]
    md += ["## Failures and errors", "", _md_table(fhead, failures) if failures else "None.", ""]
    md_path = out_dir / "report.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    parts = [f"<h1>Evaluation report</h1><p>{len(runs)} task runs.</p>",
             "<h2>Summary</h2>", _html_table(shead, srows), "<h2>Tasks</h2>", _html_table(thead, trows)]
    if xrows:
        parts += ["<h2>Exploration</h2>", _html_table(xhead, xrows)]
    if regs is not None:
        parts += ["<h2>Regressions</h2>", _html_table(["task", "variant"], regs) if regs else "<p>None.</p>"]
    parts += ["<h2>Failures and errors</h2>", _html_table(fhead, failures) if failures else "<p>None.</p>"]
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width,initial-scale=1"><title>Evaluation report</title><style>{_CSS}</style>'
            f'</head><body>{"".join(parts)}</body></html>')
    html_path = out_dir / "report.html"
    html_path.write_text(page, encoding="utf-8")
    return md_path, html_path
