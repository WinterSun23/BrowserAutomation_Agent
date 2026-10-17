from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher

from ..embed import cosine
from ..llm import complete_json


@dataclass
class Located:
    locator: object
    method: str
    css: str | None = None


def build_locator(page, spec: str, item_match: str | None = None):
    kind, _, val = spec.partition("=")
    if kind == "role":
        role, _, name = val.partition("|name=")
        return page.get_by_role(role, name=name, exact=True) if name else page.get_by_role(role)
    if kind == "testid":
        return page.get_by_test_id(val)
    if kind == "css":
        return page.locator(val)
    if kind == "text":
        return page.get_by_text(val, exact=True)
    if kind == "item":
        item_css, _, inner = val.partition("||")
        items = page.locator(item_css)
        if item_match:
            items = items.filter(has_text=item_match)
        first = items.first
        return first.locator(inner).first if inner else first
    return None


def _ratio(a: str | None, b: str | None) -> float:
    return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()


def score_candidate(fp: dict, cand: dict) -> float:
    total = weight = 0.0

    def add(w: float, v: float) -> None:
        nonlocal total, weight
        total += w * v
        weight += w

    add(3, 1.0 if fp.get("role") == cand.get("role") else 0.0)
    add(3, _ratio(fp.get("name"), cand.get("name")))
    fa, ca = fp.get("attrs") or {}, cand.get("attrs") or {}
    for k in ("id", "name", "placeholder"):
        if fa.get(k):
            add(1, 1.0 if fa.get(k) == ca.get(k) else 0.0)
    if fa.get("class"):
        a, b = set(fa["class"].split()), set((ca.get("class") or "").split())
        add(1, len(a & b) / len(a | b) if a | b else 1.0)
    add(2, _ratio(fp.get("gcss"), cand.get("gcss")))
    if fp.get("bbox") and cand.get("bbox"):
        (x1, y1, w1, h1), (x2, y2, w2, h2) = fp["bbox"], cand["bbox"]
        d = math.dist((x1 + w1 / 2, y1 + h1 / 2), (x2 + w2 / 2, y2 + h2 / 2))
        add(1, max(0.0, 1 - d / 800))
    return total / weight if weight else 0.0


class ElementResolver:
    def __init__(self, session, embedder=None, llm=None, fp_threshold: float = 0.65, emb_threshold: float = 0.80):
        self.session, self.embedder, self.llm = session, embedder, llm
        self.fp_threshold, self.emb_threshold = fp_threshold, emb_threshold

    def resolve(self, el: dict, item_match: str | None = None, locators_only: bool = False) -> Located | None:
        """`locators_only`: only locators that name the element (role + name, exact text, test id) and nothing fuzzier,
        for optional steps where a wrong match is worse than none. A css locator could hit a different element with the
        same id (and a different, riskier name)."""
        page = self.session.page
        for i, spec in enumerate(el.get("locators") or []):
            if locators_only and not spec.startswith(("role=", "text=", "testid=")):
                continue
            try:
                loc = build_locator(page, spec, item_match)
                if loc is None:
                    continue
                n = loc.count()
                if n == 1 or (n >= 1 and spec.startswith("item=")):
                    if loc.first.is_visible():
                        return Located(loc.first, f"locator:{i}")
            except Exception:
                continue
        if locators_only:
            return None
        fp = el.get("fingerprint") or {}
        if isinstance(fp, str):
            fp = json.loads(fp)
        cands = [n for n in self.session.snapshot().nodes if n.interactive]
        if not cands:
            return None
        scored = sorted(((score_candidate(fp, asdict(c)), c) for c in cands), key=lambda t: t[0], reverse=True)
        best, second = scored[0][0], (scored[1][0] if len(scored) > 1 else 0.0)
        if best >= self.fp_threshold and best - second >= 0.05:
            return Located(page.locator(scored[0][1].css).first, "fingerprint", scored[0][1].css)
        if self.embedder:
            target = self.embedder.embed([f"{el.get('role', '')} {el.get('name_sample', '')}"])[0]
            vecs = self.embedder.embed([f"{c.role} {c.name}" for c in cands])
            sims = sorted(((cosine(target, v), c) for v, c in zip(vecs, cands)), key=lambda t: t[0], reverse=True)
            if sims[0][0] >= self.emb_threshold:
                return Located(page.locator(sims[0][1].css).first, "embedding", sims[0][1].css)
        if self.llm:
            listing = "\n".join(f"{i}. {c.role} '{c.name[:60]}'" for i, c in enumerate(cands[:60]))
            ans = complete_json(self.llm, "You match a remembered UI element to the current page.",
                                f"Remembered: {el.get('role')} '{el.get('name_sample')}'\nCandidates:\n{listing}\n"
                                'Return {"index": <number or -1>}', default={"index": -1})
            idx = ans.get("index", -1) if isinstance(ans, dict) else -1
            if isinstance(idx, int) and 0 <= idx < min(60, len(cands)):
                return Located(page.locator(cands[idx].css).first, "llm", cands[idx].css)
        return None
