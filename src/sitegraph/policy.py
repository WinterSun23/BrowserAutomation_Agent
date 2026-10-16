from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
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
