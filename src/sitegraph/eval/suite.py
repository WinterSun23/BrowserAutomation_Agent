"""Evaluation suites: YAML files listing tasks (a site, a request, optional setup requests and checks)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CHECK_KINDS = ("url_matches", "page_contains", "answer_contains_any", "result", "max_seconds", "max_tokens")
RESULT_OPS = ("==", "!=", "<=", ">=", "<", ">", "in")
# Features a task can `require`; a task needing one that is not listed here is skipped (outcome "skip").
# Add a name when the plan task that implements it lands (select: v3 Task 5, check/press: Task 6, inputs: Task 7).
SUPPORTED_FEATURES: set[str] = set()
_RESULT_STR = re.compile(r"^\s*([\w.]+)\s*(==|!=|<=|>=|<|>|in)\s*(.+?)\s*$")


@dataclass
class EvalTask:
    id: str
    site: str
    request: str
    setup: list[str]
    checks: list[dict]
    max_seconds: int = 600
    max_tokens: int | None = None
    requires: list[str] = field(default_factory=list)


@dataclass
class Suite:
    name: str
    tasks: list[EvalTask]
    defaults: dict
    serve: str | None = None      # a local directory served over HTTP for `{site}` when no --site-url is given
    variants: dict = field(default_factory=dict)   # default variants (name -> settings) when no --variant is given


def _scalar(text: str):
    """'done' -> 'done', '5' -> 5, '0.5' -> 0.5, '[a, b]' -> ['a', 'b'] (YAML rules)."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def normalize_check(spec, task_id: str) -> dict:
    """Validate one check and return it in canonical form; raises ValueError naming the task and the field."""
    if not isinstance(spec, dict) or len(spec) != 1:
        raise ValueError(f"task {task_id!r}: each check must be a mapping with exactly one kind, got {spec!r}")
    kind, value = next(iter(spec.items()))
    if kind not in CHECK_KINDS:
        raise ValueError(f"task {task_id!r}: unknown check {kind!r} (one of {', '.join(CHECK_KINDS)})")
    if kind == "url_matches":
        try:
            re.compile(str(value))
        except re.error as exc:
            raise ValueError(f"task {task_id!r}: url_matches is not a valid regex ({exc})") from None
        return {kind: str(value)}
    if kind == "page_contains":
        return {kind: str(value)}
    if kind == "answer_contains_any":
        values = value if isinstance(value, list) else [value]
        if not values:
            raise ValueError(f"task {task_id!r}: answer_contains_any needs at least one value")
        return {kind: [str(v) for v in values]}
    if kind in ("max_seconds", "max_tokens"):
        try:
            return {kind: float(value) if kind == "max_seconds" else int(value)}
        except (TypeError, ValueError):
            raise ValueError(f"task {task_id!r}: {kind} must be a number, got {value!r}") from None
    if isinstance(value, str):
        m = _RESULT_STR.match(value)
        if not m:
            raise ValueError(f"task {task_id!r}: result check {value!r} is not 'path op value'")
        value = {"path": m.group(1), "op": m.group(2), "value": _scalar(m.group(3))}
    if not isinstance(value, dict) or not {"path", "op", "value"} <= set(value):
        raise ValueError(f"task {task_id!r}: result check needs path, op and value, got {value!r}")
    if value["op"] not in RESULT_OPS:
        raise ValueError(f"task {task_id!r}: result op {value['op']!r} is not one of {' '.join(RESULT_OPS)}")
    return {"result": {"path": str(value["path"]), "op": value["op"], "value": value["value"]}}


def _fill(text: str, site_url: str | None) -> str:
    return text.replace("{site}", site_url.rstrip("/")) if site_url and "{site}" in text else text


def load_suite(path, site_url: str | None = None) -> Suite:
    """Read and validate a suite. `site_url` replaces `{site}` in task sites and requests."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict) or not isinstance(data.get("tasks"), list):
        raise ValueError(f"{path}: a suite needs a `tasks:` list")
    tasks, seen = [], set()
    for i, raw in enumerate(data["tasks"]):
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: task #{i + 1} is not a mapping")
        tid = str(raw.get("id") or "")
        if not tid:
            raise ValueError(f"{path}: task #{i + 1} has no id")
        if tid in seen:
            raise ValueError(f"{path}: duplicate task id {tid!r}")
        seen.add(tid)
        for f in ("site", "request"):
            if not raw.get(f):
                raise ValueError(f"task {tid!r}: missing field {f!r}")
        checks = raw.get("checks") or []
        if not checks:
            raise ValueError(f"task {tid!r}: needs at least one check")
        setup = raw.get("setup") or []
        if isinstance(setup, str):
            setup = [setup]
        tasks.append(EvalTask(
            id=tid, site=_fill(str(raw["site"]), site_url), request=_fill(str(raw["request"]), site_url),
            setup=[_fill(str(s), site_url) for s in setup], checks=[normalize_check(c, tid) for c in checks],
            max_seconds=int(raw.get("max_seconds", 600)),
            max_tokens=int(raw["max_tokens"]) if raw.get("max_tokens") is not None else None,
            requires=[str(r) for r in raw.get("requires") or []]))
    defaults = data.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError(f"{path}: `defaults:` must be a mapping of settings")
    variants = data.get("variants") or {}
    if not isinstance(variants, dict) or not all(isinstance(v, dict) or v is None for v in variants.values()):
        raise ValueError(f"{path}: `variants:` must map a variant name to a mapping of settings")
    return Suite(name=str(data.get("name") or Path(path).stem), tasks=tasks, defaults=dict(defaults),
                 serve=data.get("serve"), variants={str(k): dict(v or {}) for k, v in variants.items()})
