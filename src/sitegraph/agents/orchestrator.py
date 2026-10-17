from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ..browser.session import BrowserSession
from ..db import device_key, ops_key
from ..demo.director import DemoDirector
from ..graph.map_writer import MapWriter
from ..graph.queries import entry_state, recommendations
from ..migrate import rehash
from ..ids import ulid
from ..llm import complete_json
from ..normalize import domain_of
from ..ops import Ops
from ..profile import Profile
from ..schema import ensure_device_schema
from ..versions import ALGO_VERSION
from .curator import Curator
from .escalation import make_curator_escalation, make_escalation
from .navigator import ExploreReport, Navigator
from .workflows import save_workflow_from_run

KINDS = ["explore", "automate", "demo", "recommend", "save_workflow"]
MAX_NAVIGATORS = 16


@dataclass
class SiteInfo:
    domain: str
    device: str
    key: str
    ops_key: str
    entry_url: str


def _keyword_kind(text: str) -> str:
    t = text.lower()
    if re.search(r"\b(demo|video|gif|screenshot|guide|walkthrough|click-?through|record)\b", t):
        return "demo"
    if re.search(r"\b(explore|crawl|map|discover|learn the site)\b", t):
        return "explore"
    if re.search(r"\bsave\b.*\bworkflow\b", t):
        return "save_workflow"
    if re.search(r"\b(recommend|suggest|what (should|can) i)\b", t):
        return "recommend"
    return "automate"


def _clamp_agents(value) -> int | None:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return max(1, min(MAX_NAVIGATORS, n))


class Orchestrator:
    def __init__(self, settings, store, embedder, llm, blobs, events, max_workers: int = 2):
        self.s, self.store, self.embedder, self.llm, self.blobs, self.events = settings, store, embedder, llm, blobs, events
        self.profile = Profile(store, settings.uid, embedder, settings.embed_dim)
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self.session_id = self.profile.start_session()
        self._last_run: dict[str, str] = {}

    def register_site(self, url: str, device: str = "desktop") -> SiteInfo:
        domain = domain_of(url)
        info = SiteInfo(domain, device, device_key(self.s.uid, domain, device), ops_key(self.s.uid, domain), url)
        ensure_device_schema(self.store, info.key, self.s.embed_dim)
        model = getattr(self.embedder, "model_name", None) or type(self.embedder).__name__
        versions = MapWriter(self.store, info.key, self.s).ensure_site(domain, device, device, model, self.s.embed_dim)
        if versions["algo_version"] != ALGO_VERSION:
            rehash(self.store, info.key, self.blobs, self.s, domain)
        self.profile.register_site(domain, device, info.key, info.ops_key, url)
        ops = Ops(self.store, self.s.uid, domain)
        ops.register_agent(f"navigator-{device}", "navigator", device)
        for i in range(1, self.s.navigators_per_site + 1):
            ops.register_agent(f"navigator-{device}-{i}", "navigator", device)
        return info

    def site(self, domain: str, device: str = "desktop") -> SiteInfo:
        for s in self.profile.sites():
            if s["domain"] == domain and s["device"] == device:
                return SiteInfo(domain, device, s["key"], s["ops_key"], s["entry_url"])
        raise KeyError(f"unknown site {domain} ({device}); add it first")

    def classify(self, text: str) -> dict:
        default = {"kind": _keyword_kind(text), "goal": text, "params": {}}
        data = complete_json(self.llm, "You route requests for a website automation and demo agent.",
                             f"Classify this request: {text}\nReturn JSON with keys: kind (one of {KINDS}), "
                             "goal (short phrase naming the screen or outcome), params (object of input values the "
                             "user gave, e.g. {\"q\": \"red shoes\"}), item (optional list item to pick), "
                             "name (optional workflow name), navigators (optional number of parallel agents the "
                             "user asked for when exploring).", default=default)
        if not isinstance(data, dict) or data.get("kind") not in KINDS:
            return default
        data.setdefault("goal", text)
        data.setdefault("params", {})
        return data

    def submit(self, text: str, domain: str, device: str = "desktop", navigators: int | None = None) -> str:
        site = self.site(domain, device)
        ops = Ops(self.store, self.s.uid, domain)
        tid = ops.create_task("chat", text, device=device)
        self.pool.submit(self.run_task, tid, text, site, navigators)
        return tid

    # ---------- parallel exploration ----------
    def _explore(self, site: SiteInfo, policy, task_id: str, n: int) -> tuple[ExploreReport, list[str]]:
        """Run `n` Navigators on the same site graph at once. They share one scope, so every page visit and every
        click/hover path is claimed by exactly one of them (see Navigator.explore / Ops.next_item)."""
        scope = ulid("scope")
        agents = [f"navigator-{site.device}-{i}" for i in range(1, n + 1)]

        def worker(agent: str) -> ExploreReport:
            with BrowserSession(site.device, self.s.headless) as sess:
                nav = Navigator(self.store, site.key, site.domain, sess, self.s, self.embedder, self.llm, self.blobs,
                                self.events, agent_id=agent, task_id=task_id)
                return nav.explore(site.entry_url, policy, scope=scope)

        reports: list[ExploreReport] = []
        errors: list[Exception] = []
        with ThreadPoolExecutor(max_workers=n) as ex:
            for fut in [ex.submit(worker, a) for a in agents]:
                try:
                    reports.append(fut.result())
                except Exception as exc:  # one agent failing must not lose the others' work
                    errors.append(exc)
        if not reports:
            raise errors[0]
        merged = ExploreReport(run_gid=reports[0].run_gid)
        for r in reports:
            merged.states |= r.states
            merged.actions += r.actions
            merged.transitions += r.transitions
            merged.blocked += r.blocked
            merged.pages += r.pages
            merged.errors += r.errors
        merged.errors += len(errors)
        return merged, agents

    def run_task(self, task_id: str, text: str, site: SiteInfo, navigators: int | None = None) -> dict:
        ops = Ops(self.store, self.s.uid, site.domain)
        agent = f"navigator-{site.device}"
        ops.register_agent(agent, "navigator", site.device)
        ops.assign(task_id, agent)
        ops.set_status(task_id, "running")
        self.events.publish(self.s.uid, {"type": "task", "task": task_id, "status": "running", "text": text})
        policy = self.profile.default_policy()
        try:
            intent = self.classify(text)
            kind = intent["kind"]
            if kind == "demo":
                d = DemoDirector(self.store, site.key, site.domain, self.s, self.embedder, self.llm, self.blobs,
                                 self.profile, site.device)
                res = d.run(text)
                result = {"kind": kind, "storyboard": res.storyboard_gid, "renders": res.renders}
            elif kind == "recommend":
                result = {"kind": kind, **self.recommend(site)}
            elif kind == "save_workflow":
                run = self._last_run.get(site.key)
                if not run:
                    raise ValueError("nothing to save yet: run an automation first")
                wf = save_workflow_from_run(self.store, site.key, self.embedder, run,
                                            intent.get("name") or intent["goal"], intent["goal"])
                result = {"kind": kind, "workflow": wf}
            elif kind == "explore":
                # precedence: explicit argument > number named in the request > settings.navigators_per_site
                n = _clamp_agents(navigators) or _clamp_agents(intent.get("navigators")) \
                    or _clamp_agents(self.s.navigators_per_site) or 1
                for i in range(1, n + 1):
                    ops.register_agent(f"navigator-{site.device}-{i}", "navigator", site.device)
                rep, agents = self._explore(site, policy, task_id, n)
                stats = Curator(self.store, site.key, self.s, self.embedder, self.llm).run_all(
                    policy.max_risk_cost,
                    escalate=make_curator_escalation(self.s, self.store, site.key, self.embedder, self.llm))
                result = {"kind": kind, "states": len(rep.states), "actions": rep.actions,
                          "transitions": rep.transitions, "blocked": rep.blocked, "navigators": n,
                          "agents": agents, **stats}
            else:
                with BrowserSession(site.device, self.s.headless) as sess:
                    nav = Navigator(self.store, site.key, site.domain, sess, self.s, self.embedder, self.llm,
                                    self.blobs, self.events, agent_id=agent, task_id=task_id,
                                    escalate=make_escalation(self.s, self.store, site.key, site.domain, self.embedder,
                                                             self.llm, self.blobs, site.device))
                    if entry_state(self.store, site.key) is None:
                        raise ValueError("this site has not been explored yet: ask me to explore it first")
                    res = nav.execute_goal(intent["goal"], intent.get("params") or {}, policy, intent.get("item"),
                                           request_text=text)
                    self._last_run[site.key] = res.run_gid
                    Curator(self.store, site.key, self.s, self.embedder, self.llm).annotate()
                    result = {"kind": "automate", "status": res.status, "explanation": res.explanation,
                              "heals": res.heals, "replans": res.replans, "run": res.run_gid}
            ops.set_status(task_id, "done", result)
            self.profile.log_request(self.session_id, text, "done", site.key, result.get("run") or result.get("storyboard"))
            self.events.publish(self.s.uid, {"type": "task", "task": task_id, "status": "done", "result": result})
            return result
        except Exception as exc:
            result = {"kind": "error", "error": str(exc)[:500]}
            ops.set_status(task_id, "failed", result)
            self.profile.log_request(self.session_id, text, "failed", site.key)
            self.events.publish(self.s.uid, {"type": "task", "task": task_id, "status": "failed", "result": result})
            return result

    def recommend(self, site: SiteInfo) -> dict:
        return recommendations(self.store, site.key, self.profile.default_policy().max_risk_cost)
