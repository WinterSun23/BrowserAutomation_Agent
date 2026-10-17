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

    def _overlay(self, caption: str | None, box: dict | None) -> None:
        page = self.sess.page
        try:
            page.evaluate("t => window.__sg && window.__sg.caption(t)", caption or "")
            if box:
                page.evaluate("b => window.__sg && window.__sg.highlight(b[0], b[1], b[2], b[3])",
                              [box["x"], box["y"], box["width"], box["height"]])
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
        result.keyframes[shot.i] = self.sess.screenshot_png()
        if box:
            result.hotspots[shot.i] = [box["x"], box["y"], box["width"], box["height"]]

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
