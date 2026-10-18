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


def press_risk(el_in_form_submit_risk: str | None) -> str:
    """Risk of pressing a key: Enter inside a form does what the form's submit button does; any other press (Enter
    in a lone search box, Escape, arrows) at most navigates."""
    return el_in_form_submit_risk or "navigate"


_SENSITIVE_TYPES = {"password", "email", "tel"}
_SENSITIVE_AUTOCOMPLETE = re.compile(r"^(cc-.*|email|tel(-.*)?|street-address|postal-code|name|given-name|"
                                     r"family-name|username|current-password|new-password)$")
_SENSITIVE_NAME = re.compile(r"(?i)password|e-?mail|phone|card|cvv|iban|address|first.?name|last.?name|ssn")


def is_sensitive_field(el) -> bool:
    """A field that is never typed into or set from a goal or a model: passwords, e-mail, phone, payment, address
    and personal names (by input type, autocomplete hint or name). `el` is an ElementObs or an element dict."""
    get = el.get if isinstance(el, dict) else (lambda k, d=None: getattr(el, k, d))
    if (get("input_kind") or "").lower() in _SENSITIVE_TYPES:
        return True
    fp = get("fingerprint") or {}
    attrs = (fp.get("attrs") if isinstance(fp, dict) else None) or {}
    auto = str((get("limits") or {}).get("autocomplete") or "").lower().split()
    if any(_SENSITIVE_AUTOCOMPLETE.match(a) for a in auto):
        return True
    text = " ".join(str(x) for x in (get("name_norm"), get("name_sample"), attrs.get("name"), attrs.get("id"),
                                     attrs.get("placeholder")) if x)
    return bool(_SENSITIVE_NAME.search(text))
