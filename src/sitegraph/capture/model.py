from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class RawNode:
    idx: int
    role: str
    name: str
    tag: str
    href: str | None
    input_type: str | None
    parent: int | None
    depth: int
    interactive: bool
    states: dict
    bbox: list
    css: str
    gcss: str
    testid: str | None
    attrs: dict


@dataclass
class RawSnapshot:
    url: str
    title: str
    nodes: list[RawNode]

    @classmethod
    def from_dict(cls, d: dict) -> "RawSnapshot":
        return cls(url=d["url"], title=d.get("title", ""), nodes=[RawNode(**n) for n in d["nodes"]])

    def to_dict(self) -> dict:
        return {"url": self.url, "title": self.title, "nodes": [asdict(n) for n in self.nodes]}


@dataclass
class ElementObs:
    gid: str
    cs_gid: str
    role: str
    name_norm: str
    name_sample: str
    tag: str
    locators: list[str]
    fingerprint: dict
    risk: str
    risk_cost: int
    css: str = ""
    is_template: bool = False
    count: int = 1
    item_locator: str | None = None
    item_key_fields: list[str] = field(default_factory=list)
    sample_items: list[str] = field(default_factory=list)
    input_kind: str | None = None
    href_template: str | None = None
    in_form: str | None = None
    hoverable: bool = False  # likely to reveal a hidden menu on hover (aria-haspopup or a hidden sibling menu)

    def props(self) -> dict:
        d = asdict(self)
        d["fingerprint"] = json.dumps(self.fingerprint, sort_keys=True)
        return d


@dataclass
class ComponentObs:
    gid: str
    kind: str
    name: str
    blocking: bool
    ephemeral: bool
    root_css: str
    cs_gid: str
    cs_name: str
    skeleton_hash: str
    aria_state: dict
    elements: list[ElementObs]
    root_bbox: list | None = None


@dataclass
class StateObs:
    url: str
    url_template: str
    title: str
    page_gid: str
    state_gid: str
    skeleton_hash: str
    tokens: list[str]
    blocking_sig: str
    components: list[ComponentObs]
    raw: RawSnapshot | None = None

    def elements(self) -> list[ElementObs]:
        return [e for c in self.components for e in c.elements]

    def element(self, gid: str) -> ElementObs | None:
        return next((e for e in self.elements() if e.gid == gid), None)

    def component(self, gid: str) -> ComponentObs | None:
        return next((c for c in self.components if c.gid == gid), None)


def obs_to_dict(obs: StateObs) -> dict:
    d = asdict(obs)
    d.pop("raw", None)
    return d


def obs_from_dict(d: dict) -> StateObs:
    comps = [ComponentObs(**{**c, "elements": [ElementObs(**e) for e in c["elements"]]}) for c in d["components"]]
    rest = {k: v for k, v in d.items() if k not in ("components", "raw")}
    return StateObs(**rest, components=comps)
