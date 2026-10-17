from __future__ import annotations

import subprocess
import wave
from pathlib import Path

from .render import _ffmpeg


def synthesize(texts: list[str], out_dir: Path) -> list[Path]:
    """Offline TTS via pyttsx3 (optional extra: pip install pyttsx3)."""
    import pyttsx3

    out_dir.mkdir(parents=True, exist_ok=True)
    engine = pyttsx3.init()
    paths = []
    for i, t in enumerate(texts):
        p = out_dir / f"narration-{i}.wav"
        engine.save_to_file(t, str(p))
        paths.append(p)
    engine.runAndWait()
    return paths


def _duration(video: Path) -> float:
    err = subprocess.run([_ffmpeg(), "-i", str(video)], capture_output=True, text=True).stderr
    for line in err.splitlines():
        if "Duration:" in line:
            h, m, s = line.split("Duration:")[1].split(",")[0].strip().split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    return 0.0


def mix_narration(video: Path, out: Path, clips: list[tuple[float, Path]]) -> Path:
    """Place each WAV clip at its start time on a silent track matching the video, then mux."""
    if not clips:
        return video
    with wave.open(str(clips[0][1]), "rb") as w0:
        nch, width, rate = w0.getnchannels(), w0.getsampwidth(), w0.getframerate()
    total = int((_duration(video) + 0.5) * rate)
    track = bytearray(total * nch * width)
    for start, path in clips:
        with wave.open(str(path), "rb") as w:
            if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (nch, width, rate):
                continue
            frames = w.readframes(w.getnframes())
        off = int(start * rate) * nch * width
        end = min(len(track), off + len(frames))
        track[off:end] = frames[:end - off]
    audio = out.with_suffix(".wav")
    with wave.open(str(audio), "wb") as w:
        w.setnchannels(nch)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(bytes(track))
    subprocess.run([_ffmpeg(), "-y", "-i", str(video), "-i", str(audio), "-c:v", "copy", "-c:a", "aac", "-shortest",
                    str(out)], check=True, capture_output=True)
    return out
