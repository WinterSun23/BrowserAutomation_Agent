from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..graph.queries import entry_state
from ..ids import now_ms, ulid
from .capture import CaptureResult, DemoCapturer
from .render import (render_clickthrough_html, render_gif, render_guide_html, render_video,
                     render_video_segments)
from .spec import DemoSpec, interpret
from .storyboard import Storyboard, StoryboardBuilder


@dataclass
class DemoResult:
    storyboard_gid: str
    renders: list[tuple[str, str]]
    spec: DemoSpec
    status: str = "done"                 # done | needs_input | blocked | error (the Demo agent can stop early)
    answer: str | None = None            # the agent's closing message, or its question to the user


class DemoDirector:
    def __init__(self, store, key, domain, settings, embedder, llm, blobs, profile=None, device: str = "desktop"):
        self.store, self.key, self.domain, self.s = store, key, domain, settings
        self.embedder, self.llm, self.blobs, self.profile, self.device = embedder, llm, blobs, profile, device

    def run(self, text: str, start_url: str | None = None) -> DemoResult:
        prefs = self.profile.get_prefs() if self.profile else {}
        spec = interpret(self.llm, text, prefs)
        sb = StoryboardBuilder(self.store, self.key, self.embedder, self.llm).build(spec, text)
        if start_url is None:
            entry = entry_state(self.store, self.key)
            start_url = entry[1] if entry else None
        cap = DemoCapturer(self.store, self.key, self.domain, self.s, self.embedder, self.llm,
                           self.device).capture(sb, start_url)
        self._link_steps(sb, cap)
        return DemoResult(sb.gid, self.render_all(sb, cap, spec.formats), spec)

    def render_all(self, sb: Storyboard, cap: CaptureResult, formats: list[str]) -> list[tuple[str, str]]:
        """Render every format, recording a Render node (ok or failed) for each."""
        renders = []
        for fmt in formats:
            rg = ulid("rd")
            try:
                ref = self._render(fmt, sb, cap)
                status, err = "ok", None
                renders.append((fmt, ref))
            except Exception as exc:  # one failed format must not lose the others
                ref, status, err = None, "failed", str(exc)[:300]
            self.store.query(self.key, """MATCH (sb:Storyboard {gid:$sg})
                CREATE (sb)-[:RENDERED_AS]->(:Render {gid:$g, format:$f, blob_ref:$ref, status:$st, error:$e,
                                                       created:$now})""",
                             {"sg": sb.gid, "g": rg, "f": fmt, "ref": ref, "st": status, "e": err, "now": now_ms()})
        return renders

    def _link_steps(self, sb: Storyboard, cap: CaptureResult) -> None:
        for shot in sb.shots:
            sg = cap.step_gids.get(shot.i)
            if sg and getattr(shot, "gid", None):
                self.store.query(self.key, "MATCH (sh:Shot {gid:$sh}), (st:Step {gid:$st}) CREATE (sh)-[:FROM_STEP]->(st)",
                                 {"sh": shot.gid, "st": sg})

    def _slides(self, sb: Storyboard, cap: CaptureResult) -> list[tuple[str, bytes, list | None]]:
        return [(s.caption, cap.keyframes[s.i], cap.hotspots.get(s.i)) for s in sb.shots
                if not s.hidden and s.i in cap.keyframes]

    def _render(self, fmt: str, sb: Storyboard, cap: CaptureResult) -> str:
        tmp = Path(tempfile.mkdtemp(prefix="sg_render_"))
        start = cap.segments[0][0] if cap.segments else 0.0
        end = cap.segments[-1][1] if cap.segments else start + 5
        gaps = any(b[0] - a[1] > 0.5 for a, b in zip(cap.segments, cap.segments[1:]))   # skipped footage in between
        if fmt == "video":
            if gaps:
                return self.blobs.put_file(render_video_segments(cap.video_path, tmp / "demo.mp4", cap.segments))
            return self.blobs.put_file(render_video(cap.video_path, tmp / "demo.mp4", start))
        if fmt == "gif":
            if gaps:
                joined = render_video_segments(cap.video_path, tmp / "joined.mp4", cap.segments)
                total = sum(b - a for a, b in cap.segments)
                return self.blobs.put_file(render_gif(joined, tmp / "demo.gif", 0.0, min(total, 15.0)))
            return self.blobs.put_file(render_gif(cap.video_path, tmp / "demo.gif", start, min(end - start, 15.0)))
        if fmt == "screenshot":
            last = max(cap.keyframes)
            return self.blobs.put_screenshot(cap.keyframes[last])
        if fmt == "guide":
            return self.blobs.put_bytes(render_guide_html(sb.title, self._slides(sb, cap)).encode(), ".html")
        if fmt == "clickthrough":
            return self.blobs.put_bytes(render_clickthrough_html(sb.title, self._slides(sb, cap), cap.viewport).encode(),
                                        ".html")
        raise ValueError(f"unknown format {fmt}")
