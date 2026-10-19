from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

from ..agents.factory import make_navigator
from ..agents.navigator import Navigator
from ..browser.session import BrowserSession
from ..graph.queries import load_element
from .storyboard import ShotPlan, Storyboard

OVERLAY_JS = files("sitegraph.demo").joinpath("overlay.js").read_text(encoding="utf-8")


class TakeFailed(Exception):
    """A final take could not play a scene; the whole take is discarded."""

    def __init__(self, scene: int, reason: str):
        super().__init__(f"scene {scene}: {reason}")
        self.scene, self.reason = scene, reason


_ELEMENT_KINDS = {"click", "hover", "fill", "select", "check", "press"}
_SCROLL_JS = {"down": "window.scrollBy(0, window.innerHeight * 0.8)", "up": "window.scrollBy(0, -window.innerHeight * 0.8)",
              "top": "window.scrollTo(0, 0)", "bottom": "window.scrollTo(0, document.documentElement.scrollHeight)"}


@dataclass
class CaptureResult:
    video_path: Path | None
    keyframes: dict[int, bytes] = field(default_factory=dict)
    hotspots: dict[int, list] = field(default_factory=dict)
    segments: list[tuple[float, float]] = field(default_factory=list)
    run_gid: str = ""
    step_gids: dict[int, str] = field(default_factory=dict)
    viewport: tuple[int, int] = (1280, 720)
    # shot index -> (start, end) seconds in the video, for visible shots (lets a renderer pick any subset of shots)
    shot_segments: dict[int, tuple[float, float]] = field(default_factory=dict)


class DemoSession:
    """A recording browser plus the bookkeeping of what was captured. Shots are recorded one at a time, so the same
    code serves the batch capturer (a prepared storyboard) and the Demo agent (shots chosen as it goes)."""

    def __init__(self, store, key, domain, settings, embedder, llm, device: str = "desktop", writer=None,
                 secrets=None):
        self.store, self.key, self.domain, self.s = store, key, domain, settings
        self.embedder, self.llm, self.device = embedder, llm, device
        self.writer, self.secrets = writer, secrets
        self.result = CaptureResult(video_path=None)
        self.sess: BrowserSession | None = None
        self.nav: Navigator | None = None
        self.run: str | None = None
        self.cur: str | None = None

    @property
    def is_open(self) -> bool:
        return self.sess is not None

    def open(self, start_url: str) -> None:
        video_dir = Path(tempfile.mkdtemp(prefix="sg_video_"))
        self.sess = BrowserSession(self.device, self.s.headless, record_video_dir=video_dir)
        self.sess.__enter__()
        self.sess.context.add_init_script(OVERLAY_JS)
        vp = self.sess.page.viewport_size or {"width": 1280, "height": 720}
        self.result.viewport = (vp["width"], vp["height"])
        self.nav = make_navigator(self.store, self.key, self.domain, self.sess, self.s, self.embedder, self.llm,
                                  agent_id="demo-director", writer=self.writer, secrets=self.secrets)
        self.sess.goto(start_url)
        self.run = self.nav.activity.start_run("demo-capture", "demo-director", request=start_url)
        self.result.run_gid = self.run
        self.cur = self.nav.current_state()[0]

    def _overlay(self, caption: str | None, box: dict | None, zoom: bool = False) -> None:
        page = self.sess.page
        try:
            page.evaluate("t => window.__sg && window.__sg.caption(t)", caption or "")
            if box:
                cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
                page.evaluate("b => window.__sg && window.__sg.highlight(b[0], b[1], b[2], b[3])",
                              [box["x"], box["y"], box["width"], box["height"]])
                page.evaluate("p => window.__sg && window.__sg.cursor(p[0], p[1])", [cx, cy])
                if zoom:
                    page.evaluate("p => window.__sg && window.__sg.zoom(p[0], p[1], 1.35)", [cx, cy])
        except Exception:
            pass

    def _highlight_and_snap(self, shot: ShotPlan, el_gid: str | None, wait_ms: int) -> None:
        """Highlight the element (if any), hold, take the keyframe."""
        result, box = self.result, None
        if el_gid:
            el = load_element(self.store, self.key, el_gid)
            loc = self.nav.resolver.resolve(el) if el else None
            box = loc.locator.bounding_box() if loc else None
        self._overlay(shot.caption, box)
        self.sess.page.wait_for_timeout(wait_ms)
        result.keyframes[shot.i] = self.sess.screenshot_png()         # taken before zooming, so the box still matches
        if box:
            result.hotspots[shot.i] = [box["x"], box["y"], box["width"], box["height"]]
            self._overlay(shot.caption, box, zoom=True)
            self.sess.page.wait_for_timeout(700)
            try:
                self.sess.page.evaluate("() => window.__sg && window.__sg.unzoom()")   # the action itself is unzoomed
            except Exception:
                pass

    def record(self, shot: ShotPlan) -> None:
        """Record one shot: highlight, keyframe, then perform its action in the browser."""
        sess, nav, result = self.sess, self.nav, self.result
        t_start = time.monotonic() - sess.started
        page = sess.page
        if shot.step is not None:
            if not shot.hidden:
                self._highlight_and_snap(shot, shot.step.el_gid, 900)
            out = nav.execute_step(shot.step, self.run, shot.i, {})
            result.step_gids[shot.i] = out.step_gid
            self.cur = out.landed
        elif shot.transition is not None:
            self._highlight_and_snap(shot, shot.transition["el_gid"], 900)
            out = nav.execute_transition(shot.transition["el_gid"], self.run, shot.i, self.cur,
                                         shot.transition.get("action_type", "click"))
            result.step_gids[shot.i] = out.step_gid
        else:  # still
            self._overlay(shot.caption, None)
            page.wait_for_timeout(700)
            result.keyframes[shot.i] = sess.screenshot_png()
        if not shot.hidden:
            page.wait_for_timeout(600)
            seg = (t_start, time.monotonic() - sess.started)
            result.segments.append(seg)
            result.shot_segments[shot.i] = seg
        self._overlay(None, None)
        try:
            page.evaluate("() => window.__sg && window.__sg.clear()")
        except Exception:
            pass

    def play(self, script, policy) -> None:
        """Record a whole script as ONE continuous take: a title card, then for every scene its step badge, explanatory
        caption and highlighted element, a hold long enough to read, the action exactly as it worked in the rehearsal and a
        moment on the result, then a closing card. Shot indices: 0 = title card, 1..n = scenes, n + 1 = closing card."""
        from .script import reading_time

        sess, page, result, n = self.sess, self.sess.page, self.result, len(script.scenes)

        def mark(i: int, t0: float) -> None:
            seg = (t0, time.monotonic() - sess.started)
            result.shot_segments[i] = seg
            if not result.segments:
                result.segments.append(seg)
            else:                                     # one continuous take: a single segment that keeps growing
                result.segments[0] = (result.segments[0][0], seg[1])

        def js(code: str, arg=None) -> None:
            try:
                page.evaluate(code, arg)
            except Exception:
                pass

        def card(i: int, title: str, text: str) -> None:
            t0 = time.monotonic() - sess.started
            js("a => window.__sg && window.__sg.card(a[0], a[1])", [title, text])
            page.wait_for_timeout(int(reading_time(f"{title} {text}") * 1000))
            result.keyframes[i] = sess.screenshot_png()
            js("() => window.__sg && window.__sg.uncard()")
            mark(i, t0)

        card(0, script.title, script.intro)
        for i, scene in enumerate(script.scenes, 1):
            t0 = time.monotonic() - sess.started
            act = scene.action or {}
            kind = act.get("kind")
            located, box = None, None
            if kind in _ELEMENT_KINDS and act.get("element"):
                if not policy.allows(act["element"]):
                    raise TakeFailed(i, f"'{act['label']}' is above the allowed risk level: demos replay safe actions only")
                located = self.nav.resolver.resolve(act["element"], item_match=act.get("item"))
                if located is None:
                    raise TakeFailed(i, f"could not find the element for '{act['label']}' on the page")
                try:                              # bring the element into view so the viewer can see what is meant
                    located.locator.scroll_into_view_if_needed(timeout=3000)
                    page.wait_for_timeout(500)
                    box = located.locator.bounding_box()
                except Exception:
                    box = None
            js("t => window.__sg && window.__sg.step(t)", f"Step {i} of {n}")
            self._overlay(scene.caption, box)
            page.wait_for_timeout(int(scene.hold * 1000))
            result.keyframes[i] = sess.screenshot_png()
            if box:
                result.hotspots[i] = [box["x"], box["y"], box["width"], box["height"]]
                if scene.zoom:
                    self._overlay(scene.caption, box, zoom=True)
                    page.wait_for_timeout(900)
                    js("() => window.__sg && window.__sg.unzoom()")
            try:
                if kind == "fill":
                    located.locator.click(timeout=5000)
                    located.locator.fill("", timeout=5000)
                    located.locator.press_sequentially(str(act.get("value", "")), delay=70)
                elif kind in _ELEMENT_KINDS:
                    self.nav._act(kind, located, act.get("value"))
                elif kind == "goto":
                    sess.goto(act["url"])
                elif kind == "scroll":
                    js(_SCROLL_JS.get(act.get("value"), _SCROLL_JS["down"]))
                elif kind == "goal":
                    self._play_route(act, i, scene.caption)
            except TakeFailed:
                raise
            except Exception as exc:
                raise TakeFailed(i, f"'{act.get('label')}' failed: {str(exc)[:200]}") from None
            if kind:
                sess.settle()
                self._overlay(scene.caption, None)
                page.wait_for_timeout(1200)           # a moment on the result of the step
            js("() => window.__sg && window.__sg.clear()")
            mark(i, t0)
        card(n + 1, script.title, script.outro or "Thanks for watching.")


    def _play_route(self, act: dict, scene: int, caption: str) -> None:
        """A rehearsal `run_goal`: replay its route step by step through the Navigator (learned steps and healing apply),
        highlighting each element as it is used."""
        from ..graph.queries import RouteStep

        self.nav._request_text = act.get("goal")
        for k, raw in enumerate(act.get("steps") or []):
            step = RouteStep(**raw)
            el = load_element(self.store, self.key, step.el_gid)
            loc = self.nav.resolver.resolve(el, item_match=act.get("item")) if el else None
            box = None
            if loc:
                try:
                    loc.locator.scroll_into_view_if_needed(timeout=3000)
                    self.sess.page.wait_for_timeout(400)
                    box = loc.locator.bounding_box()
                except Exception:
                    box = None
            self._overlay(caption, box)
            self.sess.page.wait_for_timeout(1100)
            out = self.nav.execute_step(step, self.run, scene * 100 + k, {}, act.get("item"))
            if not out.ok:
                raise TakeFailed(scene, f"step '{step.label}' of '{act.get('label')}' did not reach the expected screen")
            self.cur = out.landed

    def close(self) -> CaptureResult:
        """Stop recording (this finalises the video file) and return everything captured."""
        if self.sess is not None:
            sess, self.sess = self.sess, None
            self.nav.activity.finish_run(self.run, "done")
            sess.__exit__(None, None, None)
            self.result.video_path = sess.video_path
        return self.result


class DemoCapturer:
    """Records a prepared storyboard shot by shot in one browser session."""

    def __init__(self, store, key, domain, settings, embedder, llm, device: str = "desktop", writer=None,
                 secrets=None):
        self.store, self.key, self.domain, self.s = store, key, domain, settings
        self.embedder, self.llm, self.device = embedder, llm, device
        self.writer, self.secrets = writer, secrets

    def capture(self, sb: Storyboard, start_url: str) -> CaptureResult:
        session = DemoSession(self.store, self.key, self.domain, self.s, self.embedder, self.llm, self.device,
                              writer=self.writer, secrets=self.secrets)
        session.open(start_url)
        try:
            for shot in sb.shots:
                session.record(shot)
        finally:
            result = session.close()
        return result
