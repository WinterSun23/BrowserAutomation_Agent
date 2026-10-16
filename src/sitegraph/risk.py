from __future__ import annotations

import re

from .normalize import domain_of

RISK_COST = {"read": 0, "navigate": 0, "input": 0, "submit_safe": 1,
             "submit_mutating": 100, "destructive": 10000, "external": 1000}
RISK_ORDER = ["read", "navigate", "input", "submit_safe", "submit_mutating", "external", "destructive"]

_INPUT_ROLES = {"textbox", "searchbox", "combobox", "listbox", "spinbutton", "slider", "checkbox", "radio", "switch"}
_NAV_ROLES = {"link", "tab", "option", "menuitem", "menuitemradio", "menuitemcheckbox"}
_DESTRUCTIVE = re.compile(r"\b(delete|remove|deactivate|unsubscribe|empty cart|clear (all|cart|history)|"
                          r"cancel (order|subscription|account))\b")
_MUTATING = re.compile(r"\b(buy|pay|purchase|checkout|check out|place order|order now|subscribe|sign ?up|"
                       r"register|log ?in|sign ?in|post|send|submit|publish|donate|book now|reserve|confirm|"
                       r"follow|vote|apply now)\b")


def classify_risk(role: str, name_norm: str, href: str | None = None, page_domain: str | None = None) -> str:
    name = name_norm or ""
    if role in _INPUT_ROLES:
        return "input"
    if href:
        low = href.lower()
        if low.startswith(("mailto:", "tel:", "javascript:")):
            return "external"
        d = domain_of(href)
        if page_domain and d and d != page_domain:
            return "external"
    if _DESTRUCTIVE.search(name):
        return "destructive"
    if _MUTATING.search(name):
        return "submit_mutating"
    if role in _NAV_ROLES:
        return "navigate"
    return "submit_safe"
