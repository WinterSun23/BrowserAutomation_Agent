"""What the page itself says is wrong after an action: invalid form fields (the browser's own validation messages) and
visible error / alert text. A demo must never claim success while one of these is on screen."""
from __future__ import annotations

_JS = """() => {
  const vis = e => { const r = e.getBoundingClientRect(), s = getComputedStyle(e);
    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none'; };
  const label = e => (e.labels && e.labels[0] && e.labels[0].innerText) || e.getAttribute('aria-label') || e.name || e.id || e.type;
  const out = [];
  for (const e of document.querySelectorAll('input:invalid, select:invalid, textarea:invalid'))
    if (vis(e)) out.push(`${String(label(e)).trim().slice(0, 40)}: ${e.validationMessage}`);
  for (const e of document.querySelectorAll('[role=alert], [aria-invalid=true], .error, .field-error, .invalid-feedback'))
    if (vis(e) && (e.innerText || '').trim()) out.push((e.innerText || '').trim().replace(/\s+/g, ' ').slice(0, 120));
  return [...new Set(out)].slice(0, 5);
}"""


def page_problems(page) -> list[str]:
    try:
        return [str(p) for p in page.evaluate(_JS)]
    except Exception:
        return []
