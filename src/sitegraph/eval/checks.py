"""Checks an evaluation task run must pass (all of them)."""
from __future__ import annotations

import operator
import re
from dataclasses import dataclass

_OPS = {"==": operator.eq, "!=": operator.ne, "<=": operator.le, ">=": operator.ge, "<": operator.lt,
        ">": operator.gt, "in": lambda a, b: a in b}
_MISSING = object()


@dataclass
class CheckContext:
    result: dict
    final_url: str
    page_text: str
    answer: str
    seconds: float
    tokens: int
    metrics: dict | None = None   # the run's own numbers (states, actions, tokens, ...): read when the result lacks a path


@dataclass
class CheckResult:
    ok: bool
    detail: str


def _path(data, dotted: str):
    cur = data
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return _MISSING
    return cur


def _compare(op: str, actual, expected) -> bool:
    if op in ("<=", ">=", "<", ">"):
        return _OPS[op](float(actual), float(expected))
    if op == "in":
        return actual in (expected if isinstance(expected, (list, tuple, set, str)) else [expected])
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return _OPS[op](float(actual), float(expected))
    return _OPS[op](str(actual), str(expected))


def check(spec: dict, ctx: CheckContext) -> CheckResult:
    """One check (canonical form from `suite.normalize_check`, short result strings are accepted too)."""
    from .suite import normalize_check

    spec = normalize_check(spec, "?")
    kind, value = next(iter(spec.items()))
    if kind == "url_matches":
        ok = re.search(value, ctx.final_url or "") is not None
        return CheckResult(ok, f"url {ctx.final_url!r} {'matches' if ok else 'does not match'} /{value}/")
    if kind == "page_contains":
        ok = value.lower() in (ctx.page_text or "").lower()
        return CheckResult(ok, f"page {'contains' if ok else 'does not contain'} {value!r}")
    if kind == "answer_contains_any":
        ans = (ctx.answer or "").lower()
        hit = next((v for v in value if v.lower() in ans), None)
        return CheckResult(hit is not None, f"answer contains {hit!r}" if hit is not None else
                           f"answer {ctx.answer[:120]!r} contains none of {value}")
    if kind == "max_seconds":
        return CheckResult(ctx.seconds <= value, f"{ctx.seconds:.1f}s (max {value:g}s)")
    if kind == "max_tokens":
        return CheckResult(ctx.tokens <= value, f"{ctx.tokens} tokens (max {value})")
    path, op, expected = value["path"], value["op"], value["value"]
    actual = _path(ctx.result, path)
    if actual is _MISSING and ctx.metrics:
        actual = _path(ctx.metrics, path)
    if actual is _MISSING:
        return CheckResult(False, f"result has no {path}")
    try:
        ok = _compare(op, actual, expected)
    except (TypeError, ValueError):
        ok = False
    return CheckResult(ok, f"result {path} = {actual!r}, expected {op} {expected!r}")
