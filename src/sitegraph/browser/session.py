from __future__ import annotations

import os
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
_SCROLL_JS = """async () => {
  const de = document.documentElement;
  if (de.scrollHeight <= window.innerHeight + 4) return false;
  const h0 = de.scrollHeight;
  window.scrollTo(0, h0);
  await new Promise(r => setTimeout(r, 250));
  return de.scrollHeight > h0;
}"""
_LISTENERS_JS = """(() => {
  const types = new Set(['click', 'dblclick', 'mousedown', 'mouseup', 'pointerdown', 'pointerup', 'touchstart',
                         'mouseover', 'mouseenter']);
  const found = new WeakSet();
  for (const el of document.querySelectorAll('*')) {
    let ls; try { ls = getEventListeners(el); } catch (e) { continue; }
    for (const t in ls) { if (types.has(t) && ls[t].length) { found.add(el); break; } }
  }
  window.__sgClickable = found;
  return true;
})()"""
_DEVICES = {"phone": "iPhone 13", "tablet": "iPad Pro 11"}


class BrowserSession:
    def __init__(self, device: str = "desktop", headless: bool = True,
                 record_video_dir: Path | None = None, video_size: tuple[int, int] = (1280, 720),
                 detect_listeners: bool | None = None):
        self.device, self.headless = device, headless
        # find role-less clickables through the page's own event listeners (Chromium only; SITEGRAPH_NO_LISTENERS=1 turns it off)
        self.detect_listeners = (not os.environ.get("SITEGRAPH_NO_LISTENERS")) if detect_listeners is None else detect_listeners
        self._cdp, self._cdp_page = None, None
        self.record_video_dir, self.video_size = record_video_dir, video_size
        self.video_path: Path | None = None

    def __enter__(self) -> "BrowserSession":
        self._pw = sync_playwright().start()
        # an installed browser ("chrome", "msedge") instead of Playwright's own Chromium, when set
        channel = os.environ.get("SITEGRAPH_BROWSER_CHANNEL") or None
        self.browser = self._pw.chromium.launch(headless=self.headless, channel=channel)
        args: dict = dict(self._pw.devices[_DEVICES[self.device]]) if self.device in _DEVICES else \
            {"viewport": {"width": 1280, "height": 720}}
        args.pop("default_browser_type", None)  # a device-descriptor field that new_context() does not accept
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

    def load_all(self, max_scrolls: int = 8) -> int:
        """Scroll down in steps until the page stops growing (lazy / infinite loading), then back to the top.
        Short pages return immediately. Returns how many scroll steps loaded more content."""
        grown = 0
        try:
            for _ in range(max_scrolls):
                if not self.page.evaluate(_SCROLL_JS):
                    break
                grown += 1
                self.settle(150)
            self.page.evaluate("window.scrollTo(0, 0)")
        except Exception:
            pass  # page navigated mid-scroll; the next snapshot re-reads the DOM
        return grown

    def _mark_listeners(self) -> None:
        """Remember (in the page, in a WeakSet) every element with a click / pointer / hover listener, using the
        DevTools command-line API `getEventListeners`, which plain page scripts cannot call. One CDP call, no cap on
        the number of elements. Any failure (another browser engine, a page that navigated meanwhile) just means
        the cheap hints alone are used."""
        try:
            if self._cdp is None or self._cdp_page is not self.page:
                self._cdp, self._cdp_page = self.context.new_cdp_session(self.page), self.page
            self._cdp.send("Runtime.evaluate", {"expression": _LISTENERS_JS, "includeCommandLineAPI": True,
                                                "returnByValue": True})
        except Exception:
            self._cdp = None

    def snapshot(self) -> RawSnapshot:
        if self.detect_listeners:
            self._mark_listeners()
        return RawSnapshot.from_dict(self.page.evaluate(EXTRACT_JS))

    def screenshot_png(self) -> bytes:
        return self.page.screenshot(type="png")

    def frame_jpeg(self) -> bytes:
        return self.page.screenshot(type="jpeg", quality=40)

    @property
    def url(self) -> str:
        return self.page.url
