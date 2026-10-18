# Evaluation suites

The suites in this folder measure how well SiteGraph explores sites and handles requests. Each run writes
numbers you can compare across code changes, settings and models.

## Running a suite

```bash
# offline: the fixture shop served locally, no network, no model needed (the CI baseline)
python -m sitegraph eval eval/suites/offline.yaml --embedder hash

# public practice sites (needs network), two exploration modes, three repeats each
python -m sitegraph eval eval/suites/public.yaml --variant brute:exploration_mode=brute \
    --variant auto:exploration_mode=auto --repeat 3

# compare with an earlier run
python -m sitegraph eval eval/suites/public.yaml --compare data/eval/20261018-210000

# exploration modes side by side (the suite declares brute and auto variants itself)
python -m sitegraph eval eval/suites/modes.yaml --repeat 2
```

Suites:

| suite | tasks | what it measures |
|---|---|---|
| `offline.yaml` | 3 | the fixture shop: explore, a warm request, a cold request |
| `public.yaml` | 23 | practice sites: explore, warm requests (after `setup: explore`), cold requests (`-cold`) |
| `modes.yaml` | 5 | one explore task per public site, run with `exploration_mode=brute` and `auto` |

## Warm and cold tasks

A **warm** task has `setup: [explore the site]`. The map is built first, and the request is then measured on its
own. This tests target finding, routes and answers.

A **cold** task (id ending in `-cold`) has no setup. Its namespace is empty, so the request has to find its
target through goal search (`explore_toward`). Cold tasks check three things:

- the request found its target (`result: goal_explore.found == true`), plus the same URL or answer check as the
  matching warm task
- the search stayed small: `result: states <= 40` and the task's `max_tokens`. A search that finds the target
  only after crawling half the site fails.
- for searches that should use a site control first, how the target was found
  (`result: goal_explore.how == inputs`)

To run only one kind, copy the tasks you want into a small suite of your own. Each task run gets a fresh
namespace, so a warm task never benefits from a cold one or the other way round.

## Comparing exploration modes

`modes.yaml` lists `variants:` (`brute` and `auto`), so every explore task runs once per mode. `--variant` on the
command line replaces the suite's variants. Any suite can be compared the same way, for example:

```bash
python -m sitegraph eval eval/suites/public.yaml --variant brute:exploration_mode=brute \
    --variant auto:exploration_mode=auto --variant guided:exploration_mode=guided
```

The report's **Exploration** table has one row per explore task. Each variant's column shows the median
`states / actions / seconds / tokens` of its measured runs, so you can see per site whether a mode found more
screens, or the same screens with less time or fewer tokens. Without a configured model, `auto` falls back to
`brute`, so both columns should match.

Options:

- `--variant NAME:key=value,...`: settings overrides for one variant. Any `config.yaml` setting works, for
  example `exploration_mode=guided` or `llm_model=openai/gpt-4o-mini`. Repeat the flag for more variants.
  Without it, one variant named `default` runs with your configured settings.
- `--repeat N`: runs of each task per variant. Use 3 or more to see flaky tasks.
- `--out DIR`: output folder (default `data/eval/<date-time>/`).
- `--keep`: keep each task run's graphs (`u:eval-...`) for inspection. By default they are deleted.
- `--site-url URL`: fills `{site}` in a suite. The offline suite serves `tests/fixtures/site` itself when the
  flag is not given.
- `--compare DIR`: an earlier run folder. The report then shows deltas and a Regressions section.
- `-v`: progress lines (states, steps, explorers) while tasks run.

The exit code is 0 when every task passed or was skipped, 1 when one failed or errored, and 2 on bad input.

## How a task runs

Every task run gets its own uid namespace (`eval-<run>-<variant>-<task>-<rep>`), so runs never share a map.
The runner first runs the task's `setup` requests (usually `explore the site`), then its `request`, and then
evaluates its `checks` against the result, the final page and the answer. All checks must pass.

Check kinds:

| check | passes when |
|---|---|
| `url_matches: REGEX` | the final page URL matches the regex |
| `page_contains: TEXT` | the final page text contains TEXT (case-insensitive) |
| `answer_contains_any: [A, B]` | the answer of an `ask` request contains one of the values |
| `result: PATH OP VALUE` | a field of the request result compares true, e.g. `states >= 10`, `status == done`, `goal_explore.found == true` (dotted paths; ops `== != <= >= < > in`). A path the result lacks is read from the run's own numbers (`states`, `actions`, `tokens`, `llm_calls`, `heals`, `replans`, ...). If neither has it, the check fails with "result has no PATH". |
| `max_seconds: N` | the request took at most N seconds |
| `max_tokens: N` | the run used at most N model tokens |

A task's `max_seconds` field (default 600) is also a hard timeout. A task with `requires: [select]` is skipped
until that feature is listed in `sitegraph.eval.suite.SUPPORTED_FEATURES`.

## Reading the report

Each run folder has:

- `results.jsonl`: one line per task run, written as each run ends.
- `report.md` and `report.html`: the same content. The HTML file needs no network.

Outcomes:

- `pass`: every check passed.
- `fail`: a check failed, the request failed, or the task timed out. This is a problem with SiteGraph.
- `error`: the environment failed. Examples are a site that is down, a network error, or a browser that could
  not start. Errors are counted on their own and are not in the pass rate.
- `skip`: the task needs a feature that is not built yet.

The summary shows, per variant:

- the pass rate (pass / (pass + fail))
- the median seconds and tokens of the measured runs
- the error and skip counts
- the flaky tasks: tasks that both passed and failed across repeats

The task table shows `passes/runs` for each task and variant. With `--compare`, the Regressions section lists the
tasks that passed every repeat in the baseline but failed at least once now.

## Etiquette for public sites

- Keep budgets small. The public suite caps exploration at 40 states and 180 seconds per task (`defaults.policy`).
- Exploration respects `robots.txt` (`respect_robots` in the policy). Do not turn that off for these sites.
- Do not run many repeats in a loop against the same site. Three repeats of the suite are plenty.
- The suite never logs in or types personal data. Sensitive fields are never filled, whatever a task says.
