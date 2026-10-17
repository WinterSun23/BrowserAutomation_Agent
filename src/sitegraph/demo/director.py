from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..graph.queries import entry_state
from ..graph.recording import copy_activity
from ..ids import now_ms, ulid
from .capture import CaptureResult, DemoCapturer
from .render import (render_clickthrough_html, render_gif, render_guide_html, render_guide_markdown,
                     render_video, render_video_segments)
from .spec import DemoSpec, interpret
from .storyboard import Storyboard, StoryboardBuilder, load_storyboard


@dataclass
class DemoResult:
    storyboard_gid: str
    renders: list[tuple[str, str]]
    spec: DemoSpec
    status: str = "done"                 # done | needs_input | blocked | error (the Demo agent can stop early)
    answer: str | None = None            # the agent's closing message, or its question to the user


class DemoDirector:
    def __init__(self, store, key, domain, settings, embedder, llm, blobs, profile=None, device: str = "desktop",
                 git=None):
        self.store, self.key, self.domain, self.s = store, key, domain, settings
        self.embedder, self.llm, self.blobs, self.profile, self.device = embedder, llm, blobs, profile, device
        self.git = git                       # GitOps: demo captures are committed and storyboards pinned to the map

    def run(self, text: str, start_url: str | None = None) -> DemoResult:
        prefs = self.profile.get_prefs() if self.profile else {}
        memories = [t for t, sc in self.profile.recall(text, 3) if sc >= 0.5] if self.profile else None
        spec = interpret(self.llm, text, prefs, memories)
        sb = StoryboardBuilder(self.store, self.key, self.embedder, self.llm).build(
            spec, text, map_commit=self.git.head(self.device) if self.git else None)
        if start_url is None:
            entry = entry_state(self.store, self.key)
            start_url = entry[1] if entry else None
        cap = self._capture(self.key, sb, start_url)
        self._link_steps(sb, cap)
        return DemoResult(sb.gid, self.render_all(sb, cap, self._with_markdown(spec.formats), spec), spec)

    @staticmethod
    def _with_markdown(formats: list[str]) -> list[str]:
        """A guide also comes as Markdown + images (zip), which docs and wikis take directly."""
        return formats + ["guide_md"] if "guide" in formats and "guide_md" not in formats else list(formats)

    def _capture(self, key: str, sb: Storyboard, start_url: str) -> CaptureResult:
        """Record the storyboard. On the live map, the capture's map writes go through git (a commit on main) and the
        storyboard is pinned to the resulting version; a throwaway checkout needs neither."""
        live = self.git is not None and key == self.key
        writer = self.git.main_recorder(self.device) if live else None
        cap = DemoCapturer(self.store, key, self.domain, self.s, self.embedder, self.llm, self.device,
                           writer=writer).capture(sb, start_url)
        if live:
            self.git.commit_main(self.device, writer, "demo-director", f"demo capture {sb.gid}")
            self.pin(sb)
        return cap

    def pin(self, sb: Storyboard) -> None:
        """Record which version of the map this storyboard was made against."""
        if self.git is None:
            return
        sb.map_commit = self.git.head(self.device)
        self.store.query(self.key, "MATCH (s:Storyboard {gid:$g}) SET s.map_commit = $c", {"g": sb.gid, "c": sb.map_commit})

    def rerender(self, sb_gid: str, pinned: bool = False) -> DemoResult:
        """Record and render a saved storyboard again. `pinned`: against the map as it was when the demo was made
        (a throwaway checkout of its commit); otherwise against the map as it is now."""
        sb = load_storyboard(self.store, self.key, sb_gid)
        spec_json = self.store.ro_query(self.key, "MATCH (r:DemoRequest)-[:PRODUCED]->(:Storyboard {gid:$g}) "
                                                  "RETURN r.interpretation", {"g": sb_gid})[0][0]
        spec = DemoSpec(**json.loads(spec_json))
        if not spec.formats:         # a storyboard the Demo agent composed stores no formats: reuse what it rendered
            spec.formats = [f for (f,) in self.store.ro_query(
                self.key, "MATCH (:Storyboard {gid:$g})-[:RENDERED_AS]->(r:Render {status:'ok'}) "
                          "RETURN DISTINCT r.format", {"g": sb_gid})] or ["video"]
        key = self.git.checkout(self.device, sb.map_commit) if (pinned and self.git and sb.map_commit) else self.key
        entry = entry_state(self.store, key)
        if entry is None:
            raise ValueError("there is no entry screen to start the re-recording from")
        cap = self._capture(key, sb, entry[1])
        if key != self.key:
            copy_activity(self.store, key, self.key)
            self.store.delete(key)
        self._link_steps(sb, cap)
        renders = self.render_all(sb, cap, self._with_markdown(spec.formats), spec)
        self.store.query(self.key, "MATCH (s:Storyboard {gid:$g}) SET s.version = s.version + 1, s.status = 'ok'",
                         {"g": sb_gid})
        return DemoResult(sb_gid, renders, spec)

    def render_all(self, sb: Storyboard, cap: CaptureResult, formats: list[str],
                   spec: DemoSpec | None = None) -> list[tuple[str, str]]:
        """Render every format, recording a Render node (ok or failed) for each."""
        renders = []
        for fmt in formats:
            rg = ulid("rd")
            try:
                ref = self._render(fmt, sb, cap, spec)
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

    def _narrate(self, video: Path, sb: Storyboard, cap: CaptureResult, tmp: Path, gaps: bool) -> Path:
        """Speak each visible shot's narration at the moment the shot starts in the finished video."""
        from .tts import mix_narration, synthesize

        visible = [s for s in sb.shots if not s.hidden and s.i in cap.shot_segments]
        wavs = synthesize([s.narration or s.caption for s in visible], tmp / "tts")
        starts, elapsed = [], 0.0
        for s in visible:
            a, b = cap.shot_segments[s.i]
            starts.append(elapsed if gaps else a - cap.segments[0][0])    # cut footage: cumulative; contiguous: offset
            elapsed += b - a
        return mix_narration(video, tmp / "demo-narrated.mp4", list(zip(starts, wavs)))

    def _render(self, fmt: str, sb: Storyboard, cap: CaptureResult, spec: DemoSpec | None = None) -> str:
        tmp = Path(tempfile.mkdtemp(prefix="sg_render_"))
        start = cap.segments[0][0] if cap.segments else 0.0
        end = cap.segments[-1][1] if cap.segments else start + 5
        gaps = any(b[0] - a[1] > 0.5 for a, b in zip(cap.segments, cap.segments[1:]))   # skipped footage in between
        if fmt == "video":
            out = (render_video_segments(cap.video_path, tmp / "demo.mp4", cap.segments) if gaps
                   else render_video(cap.video_path, tmp / "demo.mp4", start))
            if spec is not None and spec.narration:
                try:
                    out = self._narrate(out, sb, cap, tmp, gaps)
                except Exception:             # no TTS engine installed (pip install pyttsx3): keep the silent video
                    pass
            return self.blobs.put_file(out)
        if fmt == "gif":
            if gaps:
                joined = render_video_segments(cap.video_path, tmp / "joined.mp4", cap.segments)
                total = sum(b - a for a, b in cap.segments)
                return self.blobs.put_file(render_gif(joined, tmp / "demo.gif", 0.0, min(total, 15.0)))
            return self.blobs.put_file(render_gif(cap.video_path, tmp / "demo.gif", start, min(end - start, 15.0)))
        if fmt == "screenshot":
            last = max(cap.keyframes)
            return self.blobs.put_screenshot(cap.keyframes[last])
        if fmt == "guide_md":
            return self.blobs.put_bytes(render_guide_markdown(sb.title, self._slides(sb, cap)), ".zip")
        if fmt == "guide":
            return self.blobs.put_bytes(render_guide_html(sb.title, self._slides(sb, cap)).encode(), ".html")
        if fmt == "clickthrough":
            return self.blobs.put_bytes(render_clickthrough_html(sb.title, self._slides(sb, cap), cap.viewport).encode(),
                                        ".html")
        raise ValueError(f"unknown format {fmt}")
