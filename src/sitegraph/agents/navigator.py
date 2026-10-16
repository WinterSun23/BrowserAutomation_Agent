from __future__ import annotations

import base64
import time
from collections import deque
from dataclasses import dataclass, field

from ..browser.locate import ElementResolver, Located
from ..capture.model import ElementObs, StateObs
from ..capture.segment import CLICKABLE_ROLES, absent_cs_gid, build_state
from ..graph.activity import ActivityLog
from ..graph.map_writer import MapWriter
from ..graph.queries import Route, RouteStep, best_routes, entry_state, explain, find_targets, load_element
from ..normalize import domain_of, var_name
from ..policy import ExplorationPolicy, robots_allowed


@dataclass
class ExploreReport:
    run_gid: str
    states: set = field(default_factory=set)
    actions: int = 0
    transitions: int = 0
    blocked: int = 0
    pages: int = 0
    errors: int = 0


@dataclass
class _Work:
    el_gid: str
    pre: tuple = ()          # element gids clicked first (from a fresh page load)
    pre_cstates: tuple = ()  # component states those clicks open (for ACTION.pre_cstates)


@dataclass
class StepOutcome:
    ok: bool
    landed: str
    healed: bool
    step_gid: str


@dataclass
class ExecResult:
    run_gid: str
    status: str
    route: Route | None
    landed: str | None
    replans: int
    heals: int
    explanation: str


class Navigator:
    def __init__(self, store, key, domain, session, settings, embedder, llm=None, blobs=None, events=None,
                 agent_id: str = "navigator-1", task_id: str | None = None):
        self.store, self.key, self.domain, self.session, self.s = store, key, domain, session, settings
        self.embedder, self.llm, self.blobs, self.events = embedder, llm, blobs, events
        self.agent_id, self.task_id = agent_id, task_id
        self.writer = MapWriter(store, key, settings)
        self.activity = ActivityLog(store, key)
        self.resolver = ElementResolver(session, embedder, llm)

    # ---------- observation ----------
    def observe(self) -> StateObs:
        return build_state(self.session.snapshot(), self.domain, self.s.template_min_items)

    def write_state(self, obs: StateObs) -> str:
        shot = snap = None
        if self.blobs is not None and self.writer.resolve_state(obs) is None:
            shot = self.blobs.put_screenshot(self.session.screenshot_png())
            snap = self.blobs.put_json_gz(obs.raw.to_dict()) if obs.raw else None
        return self.writer.write_state(obs, screenshot_ref=shot, snapshot_ref=snap)

    def _emit(self, event: dict, frame: bool = False) -> None:
        if self.events is None:
            return
        payload = {"task": self.task_id, **event}
        if frame:
            try:
                payload["jpeg"] = base64.b64encode(self.session.frame_jpeg()).decode()
            except Exception:
                pass
        self.events.publish(self.s.uid, payload)

    def _click(self, located: Located) -> None:
        located.locator.click(timeout=5000)
        self.session.settle()

    # ---------- exploration ----------
    def _candidates(self, obs: StateObs, policy: ExplorationPolicy, report: ExploreReport) -> list[ElementObs]:
        els = [e for c in obs.components for e in c.elements if e.role in CLICKABLE_ROLES]
        if not els:
            return []
        tried = {r[0] for r in self.store.ro_query(
            self.key, "MATCH (x:Element) WHERE x.gid IN $g AND x.tried = true RETURN x.gid",
            {"g": [e.gid for e in els]})}
        out = []
        for e in els:
            if e.gid in tried:
                continue
            if not policy.allows(e):
                report.blocked += 1
                continue
            if e.in_form and not policy.allow_forms and any(
                    x.in_form == e.in_form and x.role in ("textbox", "searchbox") for x in obs.elements()):
                continue
            out.append(e)
        return out

    def _out_of_budget(self, t0: float, policy: ExplorationPolicy, report: ExploreReport) -> bool:
        return (time.monotonic() - t0 > policy.budget_s or len(report.states) >= policy.max_states
                or report.actions + report.transitions >= policy.max_actions)

    def _replay(self, url: str, pre: tuple) -> bool:
        self.session.goto(url)
        for el_gid in pre:
            obs = self.observe()
            el = obs.element(el_gid)
            loc = self.resolver.resolve(el.props()) if el else None
            if loc is None:
                return False
            self._click(loc)
        return True

    def _fill_form(self, obs: StateObs, el: ElementObs) -> tuple[list[str], str | None]:
        inputs = [x for x in obs.elements() if x.in_form == el.in_form and x.role in ("textbox", "searchbox")]
        names = []
        for x in inputs:
            loc = self.resolver.resolve(x.props())
            if loc:
                loc.locator.fill(self.s.sample_input_text)
                names.append(var_name(x.name_norm, (x.fingerprint.get("attrs") or {}).get("name")))
        return [x.gid for x in inputs], (",".join(names) or None)

    def _diff_components(self, before: StateObs, after: StateObs) -> list[tuple[str, str, str, str, str]]:
        b = {c.gid: c for c in before.components}
        a = {c.gid: c for c in after.components}
        out = []
        for g in set(a) | set(b):
            src = b[g].cs_gid if g in b else absent_cs_gid(g)
            dst = a[g].cs_gid if g in a else absent_cs_gid(g)
            if src != dst:
                out.append((g, src, dst, b[g].cs_name if g in b else "absent", a[g].cs_name if g in a else "absent"))
        return out

    def explore(self, entry_url: str, policy: ExplorationPolicy) -> ExploreReport:
        t0 = time.monotonic()
        run = self.activity.start_run("explore", self.agent_id, self.task_id, request=f"explore {entry_url}")
        report = ExploreReport(run_gid=run)
        queue: deque[tuple[str, int]] = deque([(entry_url, 0)])
        seen_templates: set[str] = set()
        step_i, first = 0, True
        while queue and not self._out_of_budget(t0, policy, report):
            url, depth = queue.popleft()
            base_obs = None
            try:
                if policy.respect_robots and not robots_allowed(url):
                    continue
                self.session.goto(url)
                base_obs = self.observe()
            except Exception:
                report.errors += 1
                continue
            if base_obs.url_template in seen_templates:
                continue
            seen_templates.add(base_obs.url_template)
            report.pages += 1
            base = self.write_state(base_obs)
            report.states.add(base)
            if first:
                self.writer.set_entry(base)
                first = False
            work = deque(_Work(e.gid) for e in self._candidates(base_obs, policy, report))
            while work and not self._out_of_budget(t0, policy, report):
                item = work.popleft()
                try:
                    if not self._replay(url, item.pre):
                        continue
                    before = self.observe()
                    before_gid = self.write_state(before)
                    el = before.element(item.el_gid)
                    if el is None:
                        self.writer.mark_tried(item.el_gid)
                        continue
                    loc = self.resolver.resolve(el.props())
                    if loc is None:
                        self.writer.mark_tried(el.gid)
                        continue
                    requires, value_var = ([], None)
                    if el.in_form and policy.allow_forms:
                        requires, value_var = self._fill_form(before, el)
                    t_click = time.monotonic()
                    self._click(loc)
                    after = self.observe()
                except Exception as exc:  # navigation errors, detached elements, timeouts
                    report.errors += 1
                    self.writer.mark_tried(item.el_gid)
                    self.activity.add_step(run, step_i, "click", None, None, None, ok=False, error=str(exc)[:300])
                    step_i += 1
                    continue
                latency = int((time.monotonic() - t_click) * 1000)
                after_gid = self.writer.resolve_state(after) or after.state_gid
                edge_gid, dst = None, before_gid
                if after.url_template != before.url_template or after_gid != before_gid:
                    dst = self.write_state(after)
                    report.states.add(dst)
                    edge_gid = self.writer.write_action(before_gid, dst, el.props(), "click", value_var,
                                                        item.pre_cstates, requires)
                    report.actions += 1
                    if after.url_template != before.url_template:
                        if domain_of(after.url) == self.domain and depth + 1 <= policy.max_depth:
                            queue.append((after.url, depth + 1))
                    elif len(item.pre) < 2:
                        known = {e.gid for e in before.elements()}
                        for e in self._candidates(after, policy, report):
                            if e.gid not in known:
                                work.append(_Work(e.gid, item.pre + (el.gid,), item.pre_cstates))
                else:
                    changes = self._diff_components(before, after)
                    if changes:
                        self.write_state(after)
                        for comp, src, dstc, sname, dname in changes:
                            edge_gid = self.writer.write_transition(comp, src, dstc, el.props(), "click", sname, dname)
                            report.transitions += 1
                            comp_after = after.component(comp)
                            if comp_after and len(item.pre) < 2:
                                known = {e.gid for e in before.elements()}
                                for e in comp_after.elements:
                                    if e.gid not in known and e.role in CLICKABLE_ROLES and policy.allows(e):
                                        work.append(_Work(e.gid, item.pre + (el.gid,), item.pre_cstates + (dstc,)))
                self.writer.mark_tried(el.gid)
                self.activity.add_step(run, step_i, "click", edge_gid, before_gid, dst, ok=True, latency_ms=latency,
                                       value_var=value_var)
                step_i += 1
                self._emit({"type": "step", "label": f"click '{el.name_sample[:40]}'", "ok": True}, frame=True)
            self.writer.refresh_frontier(base, policy.max_risk_cost)
        for g in report.states:
            self.writer.refresh_frontier(g, policy.max_risk_cost)
        self.activity.finish_run(run, "done")
        self._emit({"type": "explore_done", "states": len(report.states), "actions": report.actions,
                    "transitions": report.transitions, "blocked": report.blocked})
        return report

    # ---------- execution ----------
    def current_state(self) -> tuple[str, StateObs]:
        if self.session.page.url in ("about:blank", ""):
            entry = entry_state(self.store, self.key)
            if entry:
                self.session.goto(entry[1])
        obs = self.observe()
        return self.write_state(obs), obs

    def _input_value(self, el: dict, params: dict) -> str:
        name = var_name(el.get("name_norm") or "", ((el.get("fingerprint") or {}).get("attrs") or {}).get("name"))
        return str(params.get(name) or params.get("query") or self.s.sample_input_text)

    def _heal_bookkeeping(self, el: dict, located: Located, step_gid: str) -> None:
        self.activity.add_observation("heal", {"method": located.method}, step_gid=step_gid, of_gid=el["gid"])
        if located.method.startswith("locator:"):
            idx = int(located.method.split(":")[1])
            self.writer.promote_locator(el["gid"], el["locators"][idx])
        elif located.css:
            self.writer.promote_locator(el["gid"], f"css={located.css}")

    def _enter_cstate(self, cs_gid: str) -> bool:
        rows = self.store.ro_query(self.key, "MATCH (:ComponentState)-[t:TRANSITION]->(:ComponentState {gid:$cs}) "
                                             "RETURN t.el_gid ORDER BY t.success_count DESC LIMIT 1", {"cs": cs_gid})
        el = load_element(self.store, self.key, rows[0][0]) if rows else None
        loc = self.resolver.resolve(el) if el else None
        if loc is None:
            return False
        self._click(loc)
        return True

    def execute_step(self, step: RouteStep, run_gid: str, i: int, params: dict,
                     item_match: str | None = None) -> StepOutcome:
        t0 = time.monotonic()
        for cs in step.pre_cstates:
            self._enter_cstate(cs)
        for in_gid in step.requires_inputs:
            inp = load_element(self.store, self.key, in_gid)
            loc = self.resolver.resolve(inp) if inp else None
            if loc:
                loc.locator.fill(self._input_value(inp, params))
        el = load_element(self.store, self.key, step.el_gid)
        located = self.resolver.resolve(el, item_match=item_match) if el else None
        if located is None:
            self.writer.record_attempt(step.action_gid, False)
            sg = self.activity.add_step(run_gid, i, step.action_type, step.action_gid, step.src, None, ok=False,
                                        error="element not found", value_var=step.value_var)
            self._emit({"type": "step", "label": step.label, "ok": False})
            return StepOutcome(False, step.src, False, sg)
        healed = located.method != "locator:0"
        try:
            self._click(located)
        except Exception as exc:
            self.writer.record_attempt(step.action_gid, False)
            sg = self.activity.add_step(run_gid, i, step.action_type, step.action_gid, step.src, None, ok=False,
                                        error=str(exc)[:300], value_var=step.value_var)
            return StepOutcome(False, step.src, healed, sg)
        obs = self.observe()
        landed = self.write_state(obs)
        ok = landed == step.dst
        if ok:
            self.writer.write_action(step.src, step.dst, el, step.action_type, step.value_var, step.pre_cstates,
                                     step.requires_inputs, success=True)
        else:
            self.writer.record_attempt(step.action_gid, False)
            self.writer.write_action(step.src, landed, el, step.action_type, step.value_var, step.pre_cstates,
                                     step.requires_inputs, success=True)
        sg = self.activity.add_step(run_gid, i, step.action_type, step.action_gid, step.src, landed, ok=ok,
                                    healed=healed, heal_method=located.method if healed else None,
                                    latency_ms=int((time.monotonic() - t0) * 1000), value_var=step.value_var)
        if healed:
            self._heal_bookkeeping(el, located, sg)
        if not ok:
            self.activity.add_observation("unexpected_state", {"expected": step.dst, "landed": landed}, step_gid=sg)
        self._emit({"type": "step", "label": step.label, "ok": ok, "healed": healed}, frame=True)
        return StepOutcome(ok, landed, healed, sg)

    def execute_transition(self, el_gid: str, run_gid: str, i: int, from_state: str) -> StepOutcome:
        el = load_element(self.store, self.key, el_gid)
        located = self.resolver.resolve(el) if el else None
        if located is None:
            sg = self.activity.add_step(run_gid, i, "click", None, from_state, None, ok=False, error="element not found")
            return StepOutcome(False, from_state, False, sg)
        self._click(located)
        sg = self.activity.add_step(run_gid, i, "click", None, from_state, from_state, ok=True,
                                    healed=located.method != "locator:0")
        return StepOutcome(True, from_state, located.method != "locator:0", sg)

    def drive_to(self, target_gid: str, run_gid: str, counter: list[int], params: dict, policy: ExplorationPolicy,
                 item_match: str | None = None) -> tuple[str, str, Route | None, int, int]:
        cur, _ = self.current_state()
        replans = heals = 0
        last: Route | None = None
        while cur != target_gid:
            routes = best_routes(self.store, self.key, cur, target_gid, policy.max_risk_cost)
            if not routes:
                return "no_route", cur, last, replans, heals
            last = routes[0]
            self._emit({"type": "route", "actions": [s.action_gid for s in last.steps],
                        "states": [last.steps[0].src] + [s.dst for s in last.steps] if last.steps else []})
            for step in last.steps:
                out = self.execute_step(step, run_gid, counter[0], params, item_match)
                counter[0] += 1
                heals += int(out.healed)
                cur = out.landed
                if not out.ok:
                    break
            else:
                continue
            replans += 1
            if replans > self.s.max_replans:
                return "failed", cur, last, replans, heals
        return "done", cur, last, replans, heals

    def execute_goal(self, goal: str, params: dict | None = None, policy: ExplorationPolicy | None = None,
                     item_match: str | None = None) -> ExecResult:
        policy, params = policy or ExplorationPolicy(), params or {}
        targets = [t for t in find_targets(self.store, self.key, self.embedder.embed([goal])[0])
                   if t.score >= self.s.min_target_score]
        if targets and targets[0].kind == "Workflow":
            return self.execute_workflow(targets[0].gid, params, policy, item_match)
        run = self.activity.start_run("execute", self.agent_id, self.task_id, request=goal)
        states = [t for t in targets if t.kind == "State"]
        if not states:
            self.activity.finish_run(run, "no_target")
            return ExecResult(run, "no_target", None, None, 0, 0, f"No known screen matches '{goal}'.")
        status, cur, route, replans, heals = self.drive_to(states[0].gid, run, [0], params, policy, item_match)
        self.activity.finish_run(run, status)
        text = explain(route) if route else ("Already there." if status == "done" else "No allowed route.")
        return ExecResult(run, status, route, cur, replans, heals, text)

    def execute_workflow(self, wf_gid: str, params: dict | None = None, policy: ExplorationPolicy | None = None,
                         item_match: str | None = None) -> ExecResult:
        from .workflows import load_workflow_steps

        policy, params = policy or ExplorationPolicy(), params or {}
        steps = load_workflow_steps(self.store, self.key, wf_gid)
        run = self.activity.start_run("execute", self.agent_id, self.task_id, request=f"workflow {wf_gid}")
        self.activity.link_workflow(run, wf_gid)
        counter, heals, replans, status = [0], 0, 0, "done"
        cur, _ = self.current_state()
        for st in steps:
            if st["kind"] == "TRANSITION":
                out = self.execute_transition(st["el_gid"], run, counter[0], cur)
                counter[0] += 1
                heals += int(out.healed)
                if not out.ok:
                    status = "failed"
                    break
                continue
            if cur != st["src"]:
                status, cur, _, r, h = self.drive_to(st["src"], run, counter, params, policy, item_match)
                replans, heals = replans + r, heals + h
                if status != "done":
                    break
            step = RouteStep(action_gid=st["action_gid"], label=st["label"], el_gid=st["el_gid"], src=st["src"],
                             dst=st["dst"], reliability=st["reliability"], pre_cstates=st["pre_cstates"],
                             requires_inputs=st["requires_inputs"], value_var=st["value_var"],
                             action_type=st["action_type"])
            out = self.execute_step(step, run, counter[0], params, item_match)
            counter[0] += 1
            heals += int(out.healed)
            cur = out.landed
            if not out.ok:
                status, cur, _, r, h = self.drive_to(st["dst"], run, counter, params, policy, item_match)
                replans, heals = replans + r + 1, heals + h
                if status != "done":
                    break
        self.activity.finish_run(run, status)
        ok = 1.0 if status == "done" else 0.0
        self.store.query(self.key, """MATCH (w:Workflow {gid:$w})
            SET w.run_count = coalesce(w.run_count, 0) + 1
            WITH w
            SET w.success_rate = (coalesce(w.success_rate, 0.0) * (w.run_count - 1) + $ok) / w.run_count""",
                         {"w": wf_gid, "ok": ok})
        return ExecResult(run, status, None, cur, replans, heals, f"Workflow finished with status {status}.")
