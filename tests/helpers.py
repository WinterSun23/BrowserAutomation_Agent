from __future__ import annotations

import json
import re

from sitegraph.capture.model import RawNode, RawSnapshot
from sitegraph.capture.segment import build_state
from sitegraph.normalize import domain_of

INTERACTIVE = {"link", "button", "menuitem", "textbox", "searchbox", "tab"}


def node(idx, role, name="", parent=None, depth=0, href=None, modal=False, tag="div", gcss=None, interactive=None):
    return RawNode(idx=idx, role=role, name=name, tag=tag, href=href, input_type=None, parent=parent, depth=depth,
                   interactive=(role in INTERACTIVE) if interactive is None else interactive,
                   states={"expanded": None, "selected": None, "checked": None, "disabled": "false",
                           "modal": "true" if modal else "false"},
                   bbox=[10.0 * idx, 10.0, 50.0, 20.0], css=f"#n{idx}", gcss=gcss or f"g{idx}", testid=None,
                   attrs={"id": f"n{idx}", "class": None, "name": None, "placeholder": None})


def page(url, *nodes, title="T"):
    return build_state(RawSnapshot(url=url, title=title, nodes=list(nodes)), domain_of(url))


def site_responder(system: str, user: str) -> str:
    """FakeLLM responder used by curator / navigator / demo tests."""
    m = re.search(r"Page title: (.*)", user)
    if m:
        title = m.group(1).strip()
        return json.dumps({"summary": f"{title} page", "intents": [f"open {title.lower()}"]})
    if "Name this area" in user:
        return json.dumps({"name": "Area"})
    if "Shots:" in user:
        n = len(re.findall(r"^\d+\. ", user, re.M))
        return json.dumps([f"Step {i + 1}" for i in range(n)])
    return "{}"
