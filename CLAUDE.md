# SiteGraph (BrowserAutomation_Agent)

Python web app: agents explore public websites into a FalkorDB graph, plan/execute self-healing automations, and turn graph paths into demos.

## Source of truth (all in `docs/`, which is gitignored)
- Plan to implement: `docs/plans/2026-10-16-sitegraph-v1.md` (**only v1 for now**; ignore `sitegraph-v2-full-spec.md`)
- Spec (consult when in doubt): `docs/specs/2026-10-15-graph-schema-design.md`
- Progress tracker: `docs/PROGRESS.md` — says which task is next. Read it at session start; update it after every task.

## Workflow rules (from the user)
1. Work through plan v1 one task at a time, in order.
2. For each task: first tell the user what the task is (goal, files, what it will do), then implement it following the plan's steps (tests first, as the plan specifies).
3. If there is ANY doubt (ambiguous plan step, plan vs spec conflict, failing test whose fix isn't obvious, environment choices), ask the user before deciding. Do not guess.
4. When a task is done: verify (run its tests), update `docs/PROGRESS.md`, summarise, then move to the next task.
5. `docs/` stays in `.gitignore`; never commit it.
6. **Do not git commit** (user decision). Skip the plan's commit steps and leave changes uncommitted.

## Environment
- Windows 11; Git Bash and PowerShell available. Repo root `D:\BrowserAutomation_Agent`. Docker for FalkorDB (localhost:6379), Playwright Chromium.
- Python 3.11 in `.venv`. Use **uv** for everything: `uv venv --python 3.11`, `uv pip install -e ".[dev]"`, `uv pip install <pkg>`. Run tests with `.venv/Scripts/python -m pytest`.
- The repo is already a git repo (skip the plan's `git init`).
