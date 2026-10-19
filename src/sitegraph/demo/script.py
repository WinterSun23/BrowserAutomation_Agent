"""A demo script: what the final video shows, scene by scene, written by the Demo agent after it has worked the task
out in a rehearsal. Each scene explains one step (on-screen caption, optional narration) and points at one action from
the rehearsal log, so the final take can replay exactly what worked and leave out every dead end."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

MIN_HOLD_S = 2.5          # every scene stays on screen at least this long
MAX_HOLD_S = 8.0
SECONDS_PER_WORD = 0.4    # reading time for a caption


class ScriptError(ValueError):
    pass


def reading_time(text: str) -> float:
    return min(MAX_HOLD_S, max(MIN_HOLD_S, len(text.split()) * SECONDS_PER_WORD))


@dataclass
class Scene:
    caption: str                       # on-screen text: explains the step (why), not only what is clicked
    action: dict | None = None         # one rehearsal-log entry to replay; None = a still scene (explain what is shown)
    narration: str = ""                # spoken text (voiceover); empty: the caption is spoken
    hold: float = MIN_HOLD_S           # seconds the caption stays up before the action
    zoom: bool = False                 # zoom into the element while explaining it


@dataclass
class DemoScript:
    title: str
    goal: str
    start_url: str
    intro: str
    outro: str
    scenes: list[Scene] = field(default_factory=list)

    def storyboard_caption(self, i: int) -> str:
        """The scene's text as stored in the graph: the caption only (typed values stay in the script file)."""
        return self.scenes[i].caption

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", "", f"**Goal:** {self.goal}", "", "## Intro", "", self.intro or "(none)", "",
                 "## Scenes", ""]
        for n, s in enumerate(self.scenes, 1):
            what = s.action["label"] if s.action else "show the page"
            lines.append(f"{n}. **{s.caption}**")
            lines.append(f"   - Action: {what}" + (" (zoomed in)" if s.zoom else ""))
            if s.narration:
                lines.append(f"   - Narration: {s.narration}")
            lines.append(f"   - Hold: {s.hold:.1f} s")
            lines.append("")
        lines += ["## Outro", "", self.outro or "(none)", ""]
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "DemoScript":
        d = json.loads(text)
        return cls(**{**d, "scenes": [Scene(**s) for s in d.get("scenes") or []]})


def build_script(*, title: str, goal: str, start_url: str, intro: str, outro: str, scenes: list[dict],
                 log: list[dict]) -> DemoScript:
    """Turn the agent's scene list into a script. Scenes name rehearsal actions by id ('a3'); unknown ids are an
    error the agent can fix, so a script can only replay what really happened in the rehearsal."""
    if not scenes:
        raise ScriptError("a script needs at least one scene")
    by_id = {e["id"]: e for e in log}
    out = []
    for n, s in enumerate(scenes, 1):
        caption = str(s.get("caption") or "").strip()
        if not caption:
            raise ScriptError(f"scene {n} has no caption")
        action = None
        if s.get("action"):
            if s["action"] not in by_id:
                known = ", ".join(f"{e['id']} ({e['label']})" for e in log) or "none"
                raise ScriptError(f"scene {n}: unknown action {s['action']}; rehearsal actions: {known}")
            action = dict(by_id[s["action"]])
        narration = str(s.get("narration") or "").strip()
        hold = s.get("hold")
        hold = max(MIN_HOLD_S, min(MAX_HOLD_S, float(hold))) if hold else reading_time(narration or caption)
        out.append(Scene(caption[:240], action, narration[:600], round(hold, 1), bool(s.get("zoom"))))
    return DemoScript(title.strip() or goal, goal, start_url, intro.strip(), outro.strip(), out)
