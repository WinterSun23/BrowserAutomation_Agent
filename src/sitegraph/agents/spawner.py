"""Start explorers as work piles up instead of all at once (so a small site does not open idle browsers)."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable


def run_scaled(agents: list[str], run_one: Callable[[str], object], backlog: Callable[[], int], *,
               min_backlog: int, after_s: float, poll_s: float = 0.5):
    """Run `run_one(agent)` for `agents[0]` now, and for each next agent only when `backlog()` has stayed at or above
    `min_backlog` for `after_s` seconds (the timer restarts after every spawn, so the next one waits for more work).
    Nothing more is started once every running agent has finished. Returns (results, errors, started) where results
    is [(agent, result)] in start order; one failing agent never loses the others' results."""
    futures: list[tuple[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, len(agents))) as ex:
        futures.append((agents[0], ex.submit(run_one, agents[0])))
        since: float | None = None
        while any(not f.done() for _, f in futures):
            if len(futures) < len(agents):
                try:
                    pending = backlog()
                except Exception:        # a failing count must not stop the exploration
                    pending = 0
                now = time.monotonic()
                if pending >= min_backlog:
                    since = now if since is None else since
                    if now - since >= after_s:
                        agent = agents[len(futures)]
                        futures.append((agent, ex.submit(run_one, agent)))
                        since = None
                else:
                    since = None
            time.sleep(poll_s)
    results, errors = [], []
    for agent, fut in futures:
        try:
            results.append((agent, fut.result()))
        except Exception as exc:
            errors.append(exc)
    return results, errors, [a for a, _ in futures]
