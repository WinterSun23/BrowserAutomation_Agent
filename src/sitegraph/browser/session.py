from __future__ import annotations

import time
from importlib.resources import files
from pathlib import Path

from playwright.sync_api import sync_playwright

from ..capture.model import RawSnapshot

EXTRACT_JS = files("sitegraph.capture").joinpath("extract.js").read_text(encoding="utf-8")
_SETTLE_JS = """(quiet) => new Promise(res => {
  let t; const done = () => { mo.disconnect(); res(true); };
  const mo = new MutationObserver(() => { clearTimeout(t); t = setTimeout(done, quiet); });
  mo.observe(document, {subtree: true, childList: true, attributes: true});
  t = setTimeout(done, quiet); setTimeout(done, 3000);
})"""
_DEVICES = {"phone": "iPhone 13", "tablet": "iPad Pro 11"}


class BrowserSession:
    def __init__(self, device: str = "desktop", headless: bool = True,
                 record_video_dir: Path | None = None, video_size: tuple[int, int] = (1280, 720)):
        self.device, self.headless = device, headless
        self.record_video_dir, self.video_size = record_video_dir, video_size
        self.video_path: Path | None = None

    def __enter__(self) -> "BrowserSession":
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch(headless=self.headless)
        args: dict = dict(self._pw.devices[_DEVICES[self.device]]) if self.device in _DEVICES else \
            {"viewport": {"width": 1280, "height": 720}}
        if self.record_video_dir:
            args["record_video_dir"] = str(self.record_video_dir)
            args["record_video_size"] = {"width": self.video_size[0], "height": self.video_size[1]}
        self.context = self.browser.new_context(**args)
        self.page = self.context.new_page()
        self.started = time.monotonic()
        return self

    def __exit__(self, *exc) -> None:
        video = self.page.video if self.record_video_dir else None
        self.context.close()
        if video:
            self.video_path = Path(video.path())
        self.browser.close()
        self._pw.stop()

    def goto(self, url: str) -> None:
        self.page.goto(url, wait_until="domcontentloaded")
        self.settle()

    def settle(self, quiet_ms: int = 300) -> None:
        try:
            self.page.wait_for_load_state("networkidle", timeout=5000)
        except Exception:
            pass
        try:
            self.page.evaluate(_SETTLE_JS, quiet_ms)
        except Exception:
            pass  # page navigated mid-evaluation; the next snapshot re-reads the DOM

    def snapshot(self) -> RawSnapshot:
        return RawSnapshot.from_dict(self.page.evaluate(EXTRACT_JS))

    def screenshot_png(self) -> bytes:
        return self.page.screenshot(type="png")

    def frame_jpeg(self) -> bytes:
        return self.page.screenshot(type="jpeg", quality=40)

    @property
    def url(self) -> str:
        return self.page.url
