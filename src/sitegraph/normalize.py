from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlsplit

_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_HEX = re.compile(r"^[0-9a-f]{16,}$", re.I)
_NUM = re.compile(r"^\d+$")
_TRAILING_NUM = re.compile(r"^.+[-_]\d+$")
_SLUG = re.compile(r"^[a-z0-9]+(?:[-_][a-z0-9]+){2,}$", re.I)
_MONEY = re.compile(r"(?:[$€£₹¥]\s?\d[\d,]*(?:\.\d+)?)|(?:\d[\d,]*(?:\.\d+)?\s?(?:usd|eur|gbp|inr)\b)", re.I)
_DIGITS = re.compile(r"\d+")
_WS = re.compile(r"\s+")


def _template_segment(seg: str) -> str:
    base, ext = (seg.rsplit(".", 1)[0], "." + seg.rsplit(".", 1)[1]) if "." in seg else (seg, "")
    if base and any(p.match(base) for p in (_NUM, _UUID, _HEX, _TRAILING_NUM, _SLUG)):
        return ":id" + ext
    return seg


def url_template(url: str) -> str:
    parts = urlsplit(url)
    path = "/".join(_template_segment(s) for s in parts.path.split("/")) or "/"
    keys = sorted({k for k, _ in parse_qsl(parts.query, keep_blank_values=True)})
    return path + ("?" + "&".join(keys) if keys else "")


def domain_of(url: str) -> str:
    return urlsplit(url).netloc.lower()


def normalize_text(s: str | None) -> str:
    if not s:
        return ""
    t = _WS.sub(" ", s).strip().lower()
    t = _MONEY.sub("<money>", t)
    t = _DIGITS.sub("#", t)
    words = t.split(" ")
    if len(words) > 6:
        t = " ".join(words[:4]) + " …"
    return t


def sample_text(s: str | None, limit: int = 200) -> str:
    return _WS.sub(" ", s or "").strip()[:limit]


def var_name(name_norm: str, attr_name: str | None = None) -> str:
    base = (attr_name or name_norm or "").lower()
    return re.sub(r"[^a-z0-9]+", "_", base).strip("_") or "value"
