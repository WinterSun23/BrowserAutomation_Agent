import io
import json
import struct
import subprocess
import wave
import zipfile
import pytest
from sitegraph.browser.session import BrowserSession
from sitegraph.demo.capture import OVERLAY_JS
from sitegraph.demo.render import _ffmpeg, render_guide_markdown
from sitegraph.demo.spec import DemoSpec, interpret
from sitegraph.demo.storyboard import ShotPlan, StoryboardBuilder
from sitegraph.demo.tts import mix_narration
from sitegraph.llm import FakeLLM


def test_guide_markdown_zip():
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 8
    z = zipfile.ZipFile(io.BytesIO(render_guide_markdown("Demo", [("Click A", png, None), ("Done", png, None)])))
    names = set(z.namelist())
    assert {"guide.md", "images/step-1.png", "images/step-2.png"} <= names
    assert "1. Click A" in z.read("guide.md").decode()


def test_interpret_highlights_and_narration():
    spec = interpret(FakeLLM(lambda s, u: '{"formats": ["video"], "scope_kind": "highlights", "target": "main flows", '
                                          '"narration": true}'), "show me the main flows with a voiceover")
    assert spec.scope_kind == "highlights" and spec.narration is True


def test_narration_text_is_written_per_visible_shot_and_falls_back_to_captions():
    shots = [ShotPlan(0, "navigate", "open menu", True), ShotPlan(1, "navigate", "click cart", False),
             ShotPlan(2, "still", "the cart", False)]
    spec = DemoSpec(formats=["video"], scope_kind="goal", target="cart", narration=True)

    def responder(system, user):
        if "spoken narration" in system:
            return json.dumps(["Now we open the cart.", "And here it is."])
        return json.dumps(["Open the cart", "The cart"])

    StoryboardBuilder(None, None, None, FakeLLM(responder))._caption(shots, spec)
    assert [s.narration for s in shots] == ["", "Now we open the cart.", "And here it is."]
    broken = [ShotPlan(0, "still", "the cart", False)]
    StoryboardBuilder(None, None, None, FakeLLM(lambda s, u: "not json"))._caption(broken, spec)
    assert broken[0].narration == broken[0].caption != ""
    quiet = [ShotPlan(0, "still", "the cart", False)]
    StoryboardBuilder(None, None, None, FakeLLM(lambda s, u: "[]"))._caption(
        quiet, DemoSpec(formats=["video"], scope_kind="goal", target="cart"))
    assert quiet[0].narration == ""


def test_a_guide_also_comes_as_markdown():
    from sitegraph.demo.director import DemoDirector

    assert DemoDirector._with_markdown(["guide"]) == ["guide", "guide_md"]
    assert DemoDirector._with_markdown(["guide", "guide_md", "gif"]) == ["guide", "guide_md", "gif"]
    assert DemoDirector._with_markdown(["video"]) == ["video"]


def test_mix_narration(tmp_path):
    video = tmp_path / "v.mp4"
    subprocess.run([_ffmpeg(), "-y", "-f", "lavfi", "-i", "testsrc=duration=3:size=160x120:rate=10", "-pix_fmt",
                    "yuv420p", str(video)], check=True, capture_output=True)
    clip = tmp_path / "c.wav"
    with wave.open(str(clip), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"".join(struct.pack("<h", 1000 if (i // 40) % 2 else -1000) for i in range(16000)))
    out = mix_narration(video, tmp_path / "n.mp4", [(1.0, clip)])
    probe = subprocess.run([_ffmpeg(), "-i", str(out)], capture_output=True, text=True).stderr
    assert "Audio:" in probe


@pytest.mark.browser
def test_overlay_zoom_and_cursor(site_url):
    with BrowserSession() as s:
        s.context.add_init_script(OVERLAY_JS)
        s.goto(site_url + "/index.html")
        s.page.evaluate("() => { window.__sg.cursor(100, 100); window.__sg.zoom(100, 100, 1.5); }")
        assert "scale(1.5)" in s.page.evaluate("() => document.documentElement.style.transform")
        s.page.evaluate("() => window.__sg.unzoom()")
        assert s.page.evaluate("() => document.documentElement.style.transform") in ("", "none")
