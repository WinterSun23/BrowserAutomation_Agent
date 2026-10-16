from __future__ import annotations

import base64
import html
import json
import shutil
import subprocess
from pathlib import Path


def _ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def render_video(src: Path, out: Path, start: float = 0.0) -> Path:
    cmd = [_ffmpeg(), "-y", "-ss", f"{max(0.0, start):.2f}", "-i", str(src), "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return out
    except subprocess.CalledProcessError:
        fallback = out.with_suffix(".webm")
        shutil.copyfile(src, fallback)
        return fallback


def render_video_segments(src: Path, out: Path, segments: list[tuple[float, float]]) -> Path:
    """Cut the listed (start, end) segments out of the recording and join them (retakes and set-up footage dropped)."""
    cuts = [f"[0:v]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS[v{k}]" for k, (a, b) in enumerate(segments)]
    graph = ";".join(cuts) + ";" + "".join(f"[v{k}]" for k in range(len(segments))) +         f"concat=n={len(segments)}:v=1:a=0[out]"
    cmd = [_ffmpeg(), "-y", "-i", str(src), "-filter_complex", graph, "-map", "[out]", "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return out
    except subprocess.CalledProcessError:
        return render_video(src, out, segments[0][0])


def render_gif(src: Path, out: Path, start: float, duration: float, width: int = 800) -> Path:
    vf = f"fps=12,scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse"
    cmd = [_ffmpeg(), "-y", "-ss", f"{max(0.0, start):.2f}", "-t", f"{max(0.5, duration):.2f}", "-i", str(src),
           "-vf", vf, str(out)]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def _img(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


def render_guide_html(title: str, slides: list[tuple[str, bytes, list | None]]) -> str:
    items = "\n".join(f'<li><p class="cap">{i}. {html.escape(c)}</p><img src="{_img(p)}" alt="Step {i}"></li>'
                      for i, (c, p, _) in enumerate(slides, 1))
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{{font:16px/1.5 system-ui,sans-serif;max-width:960px;margin:0 auto;padding:16px;background:#fafafa;color:#111}}
ol{{list-style:none;padding:0}}li{{margin:0 0 32px}}img{{width:100%;border:1px solid #ddd;border-radius:8px}}
.cap{{font-weight:600}}</style></head><body><h1>{html.escape(title)}</h1><ol>{items}</ol></body></html>"""


def render_clickthrough_html(title: str, slides: list[tuple[str, bytes, list | None]], viewport: tuple[int, int]) -> str:
    w, h = viewport
    data = []
    for c, p, box in slides:
        hs = None
        if box:
            hs = [box[0] / w * 100, box[1] / h * 100, box[2] / w * 100, box[3] / h * 100]
        data.append({"cap": c, "img": _img(p), "hs": hs})

    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{{font:16px/1.5 system-ui,sans-serif;margin:0;padding:16px;background:#111;color:#fff}}
#stage{{position:relative;max-width:1100px;margin:0 auto}}#stage img{{width:100%;display:block;border-radius:8px}}
.hotspot{{position:absolute;border:3px solid #ff3d71;border-radius:8px;cursor:pointer;animation:p 1.2s infinite}}
@keyframes p{{50%{{box-shadow:0 0 0 8px rgba(255,61,113,.35)}}}}#cap{{max-width:1100px;margin:12px auto}}
button{{font:inherit;padding:8px 14px;border-radius:8px;border:0;cursor:pointer}}</style></head>
<body><div id="stage"><img id="img" alt=""><div id="hs" class="hotspot" hidden></div></div>
<p id="cap"></p><p style="max-width:1100px;margin:0 auto"><button id="next">Next</button></p>
<script>const S={json.dumps(data)};let i=0;const img=document.getElementById('img'),hs=document.getElementById('hs'),
cap=document.getElementById('cap'),next=document.getElementById('next');
function show(){{const s=S[i];img.src=s.img;cap.textContent=(i+1)+'/'+S.length+' — '+s.cap;
if(s.hs){{hs.hidden=false;Object.assign(hs.style,{{left:s.hs[0]+'%',top:s.hs[1]+'%',width:s.hs[2]+'%',height:s.hs[3]+'%'}});}}
else hs.hidden=true;next.textContent=i===S.length-1?'Restart':'Next';}}
function go(){{i=(i+1)%S.length;show();}}hs.onclick=go;next.onclick=go;show();</script></body></html>"""
