from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

from .risk import RISK_COST

_ROBOTS: dict[str, RobotFileParser | None] = {}


@dataclass
class ExplorationPolicy:
    name: str = "default"
    allowed_domains: list[str] = field(default_factory=list)
    max_risk: str = "submit_safe"
    max_depth: int = 3
    max_states: int = 60
    max_actions: int = 150
    budget_s: int = 300
    allow_forms: bool = False
    deny_patterns: list[str] = field(default_factory=list)
    respect_robots: bool = True

    @property
    def max_risk_cost(self) -> int:
        return RISK_COST[self.max_risk]

    def allows(self, el) -> bool:
        risk_cost = el["risk_cost"] if isinstance(el, dict) else el.risk_cost
        if risk_cost > self.max_risk_cost:
            return False
        name = el["name_sample"] if isinstance(el, dict) else el.name_sample
        href = (el.get("href_template") if isinstance(el, dict) else el.href_template) or ""
        return not any(re.search(p, f"{name} {href}", re.I) for p in self.deny_patterns)

    def with_overrides(self, overrides: dict) -> "ExplorationPolicy":
        """A copy with the given fields replaced (per-run settings from config.yaml or command-line flags)."""
        if not overrides:
            return self
        values = {}
        for key, raw in overrides.items():
            if key not in self.__dataclass_fields__ or key == "name":
                raise ValueError(f"unknown policy setting '{key}' (one of: "
                                 f"{', '.join(k for k in self.__dataclass_fields__ if k != 'name')})")
            current = getattr(self, key)
            try:
                if isinstance(current, bool):
                    if isinstance(raw, str) and raw.lower() not in ("true", "false", "1", "0", "yes", "no"):
                        raise ValueError
                    value = raw if isinstance(raw, bool) else str(raw).lower() in ("true", "1", "yes")
                elif isinstance(current, int):
                    value = int(raw)
                    if value < 0:
                        raise ValueError
                elif isinstance(current, list):
                    if not isinstance(raw, (list, tuple)) or not all(isinstance(x, str) for x in raw):
                        raise ValueError
                    value = list(raw)
                    for pattern in value if key == "deny_patterns" else []:
                        re.compile(pattern)
                else:
                    value = str(raw)
                    if key == "max_risk" and value not in RISK_COST:
                        raise ValueError
            except (ValueError, TypeError, re.error):
                hint = f" (one of: {', '.join(RISK_COST)})" if key == "max_risk" else ""
                raise ValueError(f"bad value for policy setting '{key}': {raw!r}{hint}") from None
            values[key] = value
        return replace(self, **values)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExplorationPolicy":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def robots_allowed(url: str, agent: str = "SiteGraphBot") -> bool:
    parts = urlsplit(url)
    host = parts.hostname or ""
    if host in ("127.0.0.1", "localhost") or host.endswith((".localhost", ".test")):
        return True
    base = f"{parts.scheme}://{parts.netloc}"
    if base not in _ROBOTS:
        rp = RobotFileParser(base + "/robots.txt")
        try:
            rp.read()
        except Exception:
            rp = None
        _ROBOTS[base] = rp
    rp = _ROBOTS[base]
    return True if rp is None else rp.can_fetch(agent, url)
