"""Runs an evaluation suite: every task, per variant, `repeat` times, each in a fresh uid namespace."""
from __future__ import annotations

import dataclasses
import functools
import http.server
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from ..config import Settings, _coerce
from .checks import CheckContext, check
from .suite import SUPPORTED_FEATURES, EvalTask, Suite

logger = logging.getLogger(__name__)

# failures of the environment, not of the agent: the task run is an `error`, not a `fail`
_INFRA = re.compile(r"net::ERR_|ERR_CONNECTION|connection refused|ECONNREFUSED|actively refused|getaddrinfo|"
                    r"name or service not known|nodename nor servname|BrowserType\.launch|executable doesn't exist|"
                    r"target (page, context or browser )?(has been )?closed|browser has been closed|"
                    r"browser has disconnected|cannot reach falkordb|site unreachable|ConnectionError", re.I)


@dataclass
class TaskRun:
    task_id: str
    variant: str
    rep: int
    outcome: str                      # pass | fail | error | skip
    failed: list[str] = field(default_factory=list)
    seconds: float = 0.0
    llm_calls: int = 0
    tokens: int = 0
    states: int = 0
    actions: int = 0
    heals: int = 0
    replans: int = 0
    escalations: int = 0
    spawned: int = 0
    error: str | None = None
    kind: str | None = None           # the request's kind as run (explore, automate, ask, ...)


class _CountingEvents:
    """Counts the events the metrics need (escalations, explorer spawns) and passes everything on."""

    def __init__(self, inner=None):
        self.inner, self.escalations, self.spawns = inner, 0, 0

    def channel(self, uid: str) -> str:
        return f"events:{uid}"

    def publish(self, uid: str, event: dict) -> None:
        if event.get("type") == "escalation" and event.get("status") == "started":
            self.escalations += 1
        elif event.get("type") == "spawn":
            self.spawns += 1
        if self.inner is not None:
            self.inner.publish(uid, event)


def is_infra_error(text: str | None) -> bool:
    return bool(text) and _INFRA.search(text) is not None


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")[:24] or "x"


def task_settings(base: Settings, defaults: dict, overrides: dict, uid: str) -> Settings:
    """base < suite defaults < variant overrides; values are coerced to each setting's type."""
    s = dataclasses.replace(base, uid=uid)
    names = {f.name for f in dataclasses.fields(Settings)}
    for k, v in {**defaults, **overrides}.items():
        if k not in names:
            raise ValueError(f"unknown setting {k!r} in a suite default or variant")
        cur = getattr(s, k)
        setattr(s, k, v if cur is None or isinstance(v, (dict, list)) else _coerce(v, cur))
    return s


def _reachable(url: str, timeout: float = 10.0) -> str | None:
    """None when the site answers (any status below 500), else why not."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (SiteGraph eval)"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read(256)
        return None
    except urllib.error.HTTPError as exc:
        return f"site unreachable: HTTP {exc.code}" if exc.code >= 500 else None
    except Exception as exc:
        return f"site unreachable: {type(exc).__name__}: {str(exc)[:200]}"


def _count(store, key: str, query: str) -> int:
    try:
        return int(store.ro_query(key, query)[0][0]) if store.exists(key) else 0
    except Exception:
        return 0


@functools.lru_cache(maxsize=None)
def _serve(directory: str) -> str:
    """Serve a local directory over HTTP for the rest of the process (the offline suite's fixture shop)."""
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=directory))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}"


def _with_site(suite: Suite) -> Suite:
    """Fill `{site}` placeholders from the suite's `serve:` directory when no --site-url filled them."""
    if not any("{site}" in t.site or "{site}" in t.request for t in suite.tasks):
        return suite
    if not suite.serve:
        raise ValueError("this suite uses {site}: pass --site-url (or add `serve: <directory>` to the suite)")
    url = _serve(str(Path(suite.serve).resolve()))
    fill = lambda x: x.replace("{site}", url)  # noqa: E731
    return dataclasses.replace(suite, tasks=[dataclasses.replace(t, site=fill(t.site), request=fill(t.request),
                                                                 setup=[fill(s) for s in t.setup])
                                             for t in suite.tasks])


def run_task_once(task: EvalTask, variant: str, rep: int, s: Settings, store, embedder, llm, events=None,
                  keep: bool = False) -> TaskRun:
    """One task run in the namespace `s.uid`. Never raises: environment problems give outcome `error`."""
    from ..agents.orchestrator import Orchestrator
    from ..blobs import BlobStore
    from ..ops import Ops

    run = TaskRun(task.id, variant, rep, "fail")
    missing = [r for r in task.requires if r not in SUPPORTED_FEATURES]
    if missing:
        run.outcome, run.error = "skip", f"needs unsupported feature(s): {', '.join(missing)}"
        return run
    down = _reachable(task.site)
    if down:
        run.outcome, run.error = "error", down
        return run
    counting = _CountingEvents(events)
    box: dict = {}
    orch_ref: dict = {}
    if hasattr(llm, "reset_counters"):
        llm.reset_counters()
    t0 = time.monotonic()

    def work() -> None:
        try:
            orch = Orchestrator(s, store, embedder, llm, BlobStore(s.blob_dir), counting)
            orch_ref["orch"] = orch
            site = orch.register_site(task.site)
            box["key"] = site.key
            ops = Ops(store, s.uid, site.domain)
            for text in task.setup:
                res = orch.handle(text, site).flat()
                if res.get("kind") == "error":
                    box["setup_error"] = f"setup {text!r}: {res.get('error')}"
                    return
            box["states0"] = _count(store, site.key, "MATCH (n:State) RETURN count(n)")
            box["actions0"] = _count(store, site.key, "MATCH ()-[a:ACTION]->() RETURN count(a)")
            orch.last_page = None
            box["result"] = orch.handle(task.request, site).flat()
            box["last_page"] = orch.last_page
        except Exception as exc:  # noqa: BLE001 - recorded as the run's error
            box["exception"] = f"{type(exc).__name__}: {exc}"

    th = threading.Thread(target=work, daemon=True, name=f"eval-{task.id}-{rep}")
    th.start()
    th.join(task.max_seconds)
    run.seconds = round(time.monotonic() - t0, 2)
    timed_out = th.is_alive()
    counters = llm.counters() if hasattr(llm, "counters") else {"calls": 0, "tokens": 0}
    run.llm_calls, run.tokens = counters["calls"], counters["tokens"]
    run.escalations = counting.escalations
    result = box.get("result") or {}
    key = box.get("key")
    if key and not timed_out:
        run.states = max(0, _count(store, key, "MATCH (n:State) RETURN count(n)") - box.get("states0", 0))
        run.actions = max(0, _count(store, key, "MATCH ()-[a:ACTION]->() RETURN count(a)") - box.get("actions0", 0))
    run.heals, run.replans = int(result.get("heals") or 0), int(result.get("replans") or 0)
    run.spawned = len(result["spawned"]) if isinstance(result.get("spawned"), list) else counting.spawns
    run.kind = result.get("kind")

    problem = None
    if timed_out:
        run.failed, run.error = ["timeout"], f"timeout after {task.max_seconds}s"
    elif box.get("exception") or box.get("setup_error"):
        problem = box.get("exception") or box.get("setup_error")
    elif result.get("kind") == "error":
        problem = f"request: {result.get('error')}"
    if problem:
        run.error = problem[:1000]
        if is_infra_error(problem):
            run.outcome = "error"
        else:
            run.failed = [problem[:300]]
    elif not timed_out:
        page = box.get("last_page") or {}
        answer = str(result.get("answer") or result.get("explanation") or "")
        metrics = {k: getattr(run, k) for k in ("states", "actions", "seconds", "tokens", "llm_calls", "heals",
                                                "replans", "escalations", "spawned")}
        ctx = CheckContext(result=result, final_url=page.get("url") or "", page_text=page.get("text") or "",
                           answer=answer, seconds=run.seconds, tokens=run.tokens, metrics=metrics)
        specs = list(task.checks) + ([{"max_tokens": task.max_tokens}] if task.max_tokens is not None else [])
        for spec in specs:
            c = check(spec, ctx)
            if not c.ok:
                run.failed.append(f"{next(iter(spec))}: {c.detail}")
        run.outcome = "fail" if run.failed else "pass"

    orch = orch_ref.get("orch")
    if orch is not None and not timed_out:
        try:
            orch.close()
        except Exception:  # noqa: BLE001
            pass
    if not keep:
        for k in store.list_graphs(f"u:{s.uid}:"):
            try:
                store.delete(k)
            except Exception:  # noqa: BLE001
                logger.warning("could not delete eval graph %s", k)
    return run


def run_suite(suite: Suite, variants: dict[str, dict], repeat: int, out_dir: Path, settings: Settings,
              keep: bool = False, *, llm_factory: Callable[[Settings], object] | None = None, embedder=None,
              events=None, on_run: Callable[[TaskRun], None] | None = None) -> list[TaskRun]:
    """Run every task per variant `repeat` times; `out_dir/results.jsonl` gets one line per TaskRun as it ends.
    `llm_factory(settings)` builds the model for a run (default: the configured one, like the CLI)."""
    from ..db import GraphStore
    from ..embed import HashEmbedder
    from ..llm import TracingLLM

    if llm_factory is None:
        from ..cli import make_llm as llm_factory
    suite = _with_site(suite)
    variants = variants or suite.variants or {"default": {}}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = out_dir / "results.jsonl"
    results.write_text("", encoding="utf-8")
    run_id = time.strftime("%m%d%H%M%S") + os.urandom(2).hex()
    store = GraphStore(settings.falkor_host, settings.falkor_port)
    embedder = embedder or HashEmbedder(settings.embed_dim)
    runs: list[TaskRun] = []
    for vname, overrides in variants.items():
        for rep in range(max(1, int(repeat))):
            for task in suite.tasks:
                uid = f"eval-{run_id}-{_slug(vname)}-{_slug(task.id)}-{rep}"
                try:
                    s = task_settings(settings, suite.defaults, overrides, uid)
                    llm = TracingLLM(llm_factory(s), Path(s.data_dir) / "traces")
                    tr = run_task_once(task, vname, rep, s, store, embedder, llm, events, keep)
                except Exception as exc:  # noqa: BLE001 - one broken run never stops the suite
                    tr = TaskRun(task.id, vname, rep, "error", error=f"{type(exc).__name__}: {exc}"[:1000])
                runs.append(tr)
                with open(results, "a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(tr), ensure_ascii=False) + "\n")
                if on_run:
                    on_run(tr)
    return runs
