"""Tools that look at and act on the live page. All browser access goes through here, so the risk gate cannot be
bypassed: an element whose risk class is above the active policy needs the user's one-time approval."""
from __future__ import annotations

import re
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

from ..browser.items import choose_item, live_item_texts
from ..db import device_key, ops_key
from ..graph.queries import entry_state
from ..graph.map_writer import MapWriter
from ..ops import Ops
from ..schema import ensure_device_schema
from ..normalize import domain_of, var_name
from ..agents.navigator import is_select
from ..risk import RISK_COST, press_risk
from ..policy import robots_allowed
from .core import ToolContext, ToolSpec, Toolbox
from .graph_tools import policy_of

MAX_ELEMENTS = 150


def get_nav(ctx: ToolContext):
    """The Navigator for the site the browser is on now (its resolver, writer and settle helpers), on the context's
    single browser. Pages of another site than the session's get a Navigator of their own, so they are judged
    against their own domain (its links are not 'external') and recorded in that site's own graph."""
    home = _home_nav(ctx)
    page = ctx.session.page.url if ctx.session is not None else ""
    domain = domain_of(page) if page.startswith("http") else ""
    if not domain or domain == ctx.domain:
        return home
    navs = ctx.extra.setdefault("domain_navs", {})
    if domain not in navs:
        navs[domain] = _site_nav(ctx, domain, page)
    return navs[domain]


def _site_nav(ctx: ToolContext, domain: str, url: str):
    """A Navigator on `domain`'s own graph (created and registered the first time the agent visits that site)."""
    from ..agents.factory import make_navigator

    s, device = ctx.settings, ctx.device
    key = device_key(s.uid, domain, device)
    ensure_device_schema(ctx.store, key, s.embed_dim)
    model = getattr(ctx.embedder, "model_name", None) or type(ctx.embedder).__name__
    MapWriter(ctx.store, key, s).ensure_site(domain, device, device, model, s.embed_dim)
    if ctx.profile is not None:
        ctx.profile.register_site(domain, device, key, ops_key(s.uid, domain), url)
    return make_navigator(ctx.store, key, domain, ctx.browser(), s, ctx.embedder, ctx.llm, blobs=ctx.blobs,
                          agent_id="chat-agent", ops=Ops(ctx.store, s.uid, domain),
                          secrets=(lambda var: ctx.profile.secret_for(var, domain)) if ctx.profile else None)


def _home_nav(ctx: ToolContext):
    nav = ctx.extra.get("nav")
    if nav is None:
        from ..agents.escalation import make_escalation
        from ..agents.factory import make_navigator

        nav = make_navigator(ctx.store, ctx.key, ctx.domain, ctx.browser(), ctx.settings, ctx.embedder, ctx.llm,
                             blobs=ctx.blobs, agent_id="chat-agent", ops=ctx.ops, writer=ctx.writer,
                             secrets=(lambda var: ctx.profile.secret_for(var, ctx.domain)) if ctx.profile else None,
                             escalate=make_escalation(ctx.settings, ctx.store, ctx.key, ctx.domain, ctx.embedder,
                                                      ctx.llm, ctx.blobs, ctx.device))
        ctx.extra["nav"] = nav
    return nav


def _variants(value: str) -> list[str]:
    v = value.strip()
    out = {v, quote(v), v.replace(" ", "+"), v.replace(" ", "%20"), v.replace(" ", "-"), v.replace(" ", "_")}
    return sorted((x for x in out if len(x) >= 2), key=len, reverse=True)


def _scrub_str(text: str | None, typed: list[str]) -> str | None:
    if not text:
        return text
    for value in typed:
        for var in _variants(value):
            text = re.sub(re.escape(var), "<input>", text, flags=re.I)
    return text


def _scrub_url(url: str, typed: list[str]) -> str:
    parts = urlsplit(url)
    lowered = [v.strip().lower() for v in typed if len(v.strip()) >= 2]
    kept = [(k, val) for k, val in parse_qsl(parts.query, keep_blank_values=True)
            if not any(t in val.lower() for t in lowered)]
    clean = urlunsplit((parts.scheme, parts.netloc, _scrub_str(parts.path, typed), urlencode(kept), ""))
    return clean


def scrub_typed(obs, typed: list[str]):
    """Best effort: remove what the user typed from an observation before it is written to the graph or blob store
    (search terms end up in URLs, titles and result names). Text a site echoes in some other form is not caught."""
    if not typed:
        return obs
    obs.url, obs.title = _scrub_url(obs.url, typed), _scrub_str(obs.title, typed)
    obs.tokens = [_scrub_str(t, typed) for t in obs.tokens]
    if obs.raw:
        obs.raw.url, obs.raw.title = obs.url, obs.title
        for n in obs.raw.nodes:
            n.name = _scrub_str(n.name, typed)
            n.href = _scrub_url(n.href, typed) if n.href else n.href
    for e in obs.elements():
        e.name_sample, e.name_norm = _scrub_str(e.name_sample, typed), _scrub_str(e.name_norm, typed)
        e.locators = [_scrub_str(loc, typed) for loc in e.locators]
        if isinstance(e.fingerprint.get("name"), str):
            e.fingerprint["name"] = _scrub_str(e.fingerprint["name"], typed)
    return obs


def _write(ctx: ToolContext, nav, obs) -> str:
    """Write an observation to the map with anything the user typed scrubbed out first."""
    gid = nav.write_state(scrub_typed(obs, ctx.extra.get("typed", [])))
    ctx.extra.setdefault("dirty", {})[nav.key] = nav.domain      # graphs that grew: curated when the request ends
    ctx.extra["state_key"] = nav.key
    return gid


def _ensure_page(ctx: ToolContext, nav) -> None:
    if nav.session.page.url in ("about:blank", ""):
        entry = entry_state(ctx.store, ctx.key)
        if entry:
            nav.session.goto(entry[1])
        elif ctx.extra.get("entry_url"):                # an unexplored site: the URL the chat was started with
            nav.session.goto(ctx.extra["entry_url"])


def _element(ctx: ToolContext, element_id: str):
    el = ctx.extra.get("elements", {}).get(element_id)
    if el is None:
        raise ValueError(f"unknown element id '{element_id}'. Ids come from observe and change whenever the page "
                         "changes: call observe again.")
    return el


def _guard(ctx: ToolContext, el) -> None:
    if not policy_of(ctx).allows(el):
        ctx.require_approval(f"'{el.name_sample[:60]}' is classified '{el.risk}', above the allowed risk level")


def _page_summary(nav, state: str) -> dict:
    page = nav.session.page
    return {"url": page.url, "title": page.title(), "state": state}


def observe(ctx: ToolContext, args: dict):
    nav = get_nav(ctx)
    _ensure_page(ctx, nav)
    obs = nav.observe()
    state = _write(ctx, nav, obs)
    ids, listing = {}, []
    for comp in obs.components:
        for e in comp.elements:
            eid = f"e{len(ids) + 1}"
            ids[eid] = e
            if len(ids) <= MAX_ELEMENTS:
                item = {"id": eid, "role": e.role, "name": e.name_sample[:80], "in": comp.kind, "risk": e.risk}
                if e.is_template:
                    item.update(repeated_items=e.count, examples=e.sample_items[:3])
                if e.hoverable:
                    item["hoverable"] = True
                if e.options:
                    item.update(options=e.options, selected=e.selected)
                if e.checked is not None:
                    item["checked"] = e.checked
                listing.append(item)
    ctx.extra["elements"], ctx.extra["state"] = ids, state
    return {**_page_summary(nav, state), "elements": listing, "truncated": len(ids) > MAX_ELEMENTS}


def list_items(ctx: ToolContext, args: dict):
    el = _element(ctx, args["element_id"])
    if not el.is_template:
        raise ValueError("this element is not a repeated list item")
    return {"items": live_item_texts(get_nav(ctx).session.page, el.props()),
            "next": "these are all the entries with their full text (prices, ratings...): compare them yourself, then "
                    "click this element with item set to the chosen entry's text, copied exactly"}


def _after(ctx: ToolContext, nav, before: str | None, el, kind: str, value=None, value_var: str | None = None,
           props: dict | None = None) -> dict:
    """`value` is what a select / check / press did (an option label, a checked state, a key), stored on the
    action so it replays; typed text never is (`filled` only keeps variable names)."""
    before_key = ctx.extra.get("state_key")
    nav = get_nav(ctx)                       # the click may have led to another site
    obs = nav.observe()
    after = _write(ctx, nav, obs)
    filled = ctx.extra.pop("filled", {})
    if before and after != before and before_key == nav.key:    # an action never links two sites' graphs
        g = nav.writer.write_action(before, after, props or el.props(), kind,
                                    value_var or ",".join(filled.values()) or None, (), list(filled))
        nav._set_edge_value("ACTION", g, value)
    ctx.extra.pop("elements", None)
    return {**_page_summary(nav, after), "page_changed": after != before,
            "next": "call observe to see the page and get fresh element ids"}


def _resolve(ctx: ToolContext, nav, el, item: str | None):
    if el.is_template and not item and ctx.extra.get("request_text"):
        item, _ = choose_item(live_item_texts(nav.session.page, el.props()), ctx.extra["request_text"], ctx.embedder,
                              ctx.llm, ctx.settings.item_match_threshold, ctx.settings.item_match_margin)
    located = nav.resolver.resolve(el.props(), item_match=item)
    if located is None:
        raise ValueError("the element was not found on the page (it may have changed): call observe again")
    return located


def _acted(ctx: ToolContext, kind: str, el, value=None) -> None:
    """Remember what was done on the page (the recovery agent's actions can be learned as steps of a route). Actions
    above the policy are left out: an approval covers one action and must never be replayed automatically."""
    if policy_of(ctx).allows(el):
        ctx.extra.setdefault("acted", []).append(f"{kind}|{el.gid}" + (f"|{value}" if value is not None else ""))


def _what(el, item: str | None) -> str:
    """How an action is named in the rehearsal log and the script: the chosen list entry's title (without its price
    and stock text) when there is one, else the element's name."""
    text = item or el.name_sample or getattr(el, "role", "") or "control"      # an unnamed button is still a button
    return re.split(r"\s+[£$€₹¥]\s?\d", text)[0].strip()[:60]


def rehearse(ctx: ToolContext, kind: str, label: str, el=None, **data) -> None:
    """While the Demo agent rehearses, remember each action that worked, with what is needed to replay it in the final
    take (`demo/script.py`). Actions above the policy are left out: an approval is never replayed."""
    log = ctx.extra.get("rehearsal")
    if log is None:
        return
    if el is not None and not policy_of(ctx).allows(el if isinstance(el, dict) else el.props()):
        return
    entry = {"id": f"a{len(log) + 1}", "kind": kind, "label": label[:120],
             **{k: v for k, v in data.items() if v is not None}}
    if el is not None:
        entry["element"] = dict(el) if isinstance(el, dict) else el.props()
    log.append(entry)


def click(ctx: ToolContext, args: dict):
    nav, el = get_nav(ctx), _element(ctx, args["element_id"])
    _guard(ctx, el)
    located = _resolve(ctx, nav, el, args.get("item"))
    before = ctx.extra.get("state")
    nav._click(located)
    _acted(ctx, "click", el)
    rehearse(ctx, "click", f"click '{_what(el, args.get('item'))}'", el, item=args.get("item"))
    return _after(ctx, nav, before, el, "click")


def hover(ctx: ToolContext, args: dict):
    nav, el = get_nav(ctx), _element(ctx, args["element_id"])
    _guard(ctx, el)
    nav._hover(_resolve(ctx, nav, el, args.get("item")))
    _acted(ctx, "hover", el)
    rehearse(ctx, "hover", f"hover '{_what(el, args.get('item'))}'", el, item=args.get("item"))
    return {"hovered": el.name_sample[:60], "next": "call observe to see anything that appeared (menus, tooltips)"}


def fill(ctx: ToolContext, args: dict):
    nav, el = get_nav(ctx), _element(ctx, args["element_id"])
    if el.role not in ("textbox", "searchbox", "combobox"):
        raise ValueError(f"cannot type into a {el.role}")
    secret_var = args.get("from_secret")
    if (el.input_kind or "").lower() == "password" and not secret_var:
        raise ValueError("refusing to type a password: use from_secret with a stored secret")
    if secret_var:
        # the vault value is typed without ever entering the model's context, the trace or the graph
        value = ctx.profile.secret_for(secret_var, ctx.domain) if ctx.profile is not None else None
        if value is None:
            raise ValueError(f"no secret named '{secret_var}' for this site; ask the user to store it")
    elif args.get("value") is not None:
        value = args["value"]
    else:
        raise ValueError("pass either `value` (the user's own words) or `from_secret` (a stored secret's name)")
    _resolve(ctx, nav, el, None).locator.fill(value)
    name = args.get("var_name") or secret_var or var_name(el.name_norm, (el.fingerprint.get("attrs") or {}).get("name"))
    ctx.extra.setdefault("filled", {})[el.gid] = name          # only the variable NAME is remembered, never the value
    ctx.extra.setdefault("typed", []).append(value)            # in memory only: used to scrub it out of observations
    if not secret_var:                                         # a stored secret is never replayed into a recording
        rehearse(ctx, "fill", f"type into '{el.name_sample[:40]}'", el, value=value)
    return {"filled": el.name_sample[:60], "variable": name}


def _norm(text) -> str:
    return " ".join(str(text).split())


def select_option(ctx: ToolContext, args: dict):
    nav, el = get_nav(ctx), _element(ctx, args["element_id"])
    if not is_select(el):
        raise ValueError(f"element {args['element_id']} is a {el.role}, not a dropdown (<select>)")
    wanted = _norm(args["option"])
    option = (next((o for o in el.options if _norm(o) == wanted), None)
              or next((o for o in el.options if _norm(o).casefold() == wanted.casefold()), None))
    if option is None:
        raise ValueError(f"no option '{wanted}'; the options are: {', '.join(el.options)}")
    _guard(ctx, el)
    located = _resolve(ctx, nav, el, None)
    before = ctx.extra.get("state")
    nav._act("select", located, option)
    _acted(ctx, "select", el, option)
    rehearse(ctx, "select", f"choose '{option[:40]}' in '{el.name_sample[:40]}'", el, value=option)
    var = var_name(el.name_norm, (el.fingerprint.get("attrs") or {}).get("name"))
    return {"selected": option, **_after(ctx, nav, before, el, "select", option, var)}


def set_checked(ctx: ToolContext, args: dict):
    nav, el = get_nav(ctx), _element(ctx, args["element_id"])
    if el.role not in ("checkbox", "radio", "switch"):
        raise ValueError(f"element {args['element_id']} is a {el.role}, not a checkbox, radio button or switch")
    checked = args["checked"] if isinstance(args["checked"], bool) else str(args["checked"]).lower() == "true"
    _guard(ctx, el)
    located = _resolve(ctx, nav, el, None)
    before = ctx.extra.get("state")
    nav._act("check", located, str(checked))
    _acted(ctx, "check", el, str(checked))
    rehearse(ctx, "check", f"{'check' if checked else 'uncheck'} '{el.name_sample[:40]}'", el, value=str(checked))
    return {"checked": checked, **_after(ctx, nav, before, el, "check", str(checked))}


PRESS_KEYS = ("Enter", "Escape", "Tab", "ArrowDown", "ArrowUp", "PageDown", "Space")


def _form_submit_risk(ctx: ToolContext, el) -> str | None:
    """The riskiest button of the element's form (Enter there does what submitting does), or None outside forms."""
    if not el.in_form:
        return None
    buttons = [x for x in ctx.extra.get("elements", {}).values() if x.in_form == el.in_form and x.role == "button"]
    return max((b.risk for b in buttons), key=lambda r: RISK_COST[r], default=None)


def press(ctx: ToolContext, args: dict):
    key = str(args.get("key") or "").strip()
    if key not in PRESS_KEYS:
        raise ValueError(f"key '{key}' is not allowed; use one of: {', '.join(PRESS_KEYS)}")
    nav = get_nav(ctx)
    el = _element(ctx, args["element_id"]) if args.get("element_id") else None
    risk = press_risk(_form_submit_risk(ctx, el) if (el is not None and key == "Enter") else None)
    props = el.props() if el is not None else {"name_sample": "", "href_template": None}
    props.update(risk=risk, risk_cost=RISK_COST[risk])
    if not policy_of(ctx).allows(props):
        ctx.require_approval(f"pressing {key} in '{props['name_sample'][:60]}' is classified '{risk}', "
                             "above the allowed risk level")
    located = _resolve(ctx, nav, el, None) if el is not None else None
    before = ctx.extra.get("state")
    nav._act("press", located, key)
    if RISK_COST[risk] <= policy_of(ctx).max_risk_cost:
        rehearse(ctx, "press", f"press {key}" + (f" in '{props['name_sample'][:40]}'" if el is not None else ""),
                 props if el is not None else None, value=key)
    if el is None:            # a key on the page as a whole: nothing to record as an element's action
        ctx.extra.pop("elements", None)
        nav = get_nav(ctx)
        ctx.extra["state"] = _write(ctx, nav, nav.observe())
        return {"pressed": key, **_page_summary(nav, ctx.extra["state"]),
                "next": "call observe to see the page and get fresh element ids"}
    if RISK_COST[risk] <= policy_of(ctx).max_risk_cost:
        _acted(ctx, "press", el, key)
    return {"pressed": key, **_after(ctx, nav, before, el, "press", key, props=props)}


def fill_inputs(ctx: ToolContext, args: dict):
    """Set the page's search / filter controls to what the goal implies and submit (agents/inputs.py)."""
    from dataclasses import replace

    from ..agents.inputs import assign_inputs, controls_on

    nav = get_nav(ctx)
    _ensure_page(ctx, nav)
    before = ctx.extra.get("state")
    obs = nav.observe()
    assignments = assign_inputs(args["goal"], controls_on(obs), ctx.llm, ctx.embedder)
    if not assignments:
        raise ValueError("no control on this page matches the goal (search boxes, dropdowns, checkboxes, ...)")
    chosen, _, submit = nav.input_target(obs, assignments)
    policy = policy_of(ctx)
    if submit is not None and not policy.allows(submit):
        ctx.require_approval(f"submitting with '{submit.name_sample[:60]}' is classified '{submit.risk}', above "
                             "the allowed risk level")
        policy = replace(policy, max_risk_cost=max(policy.max_risk_cost, submit.risk_cost))
    ctx.extra.setdefault("typed", []).extend(str(a.value) for _, a in chosen if not isinstance(a.value, bool))
    landed, how = nav.apply_inputs(obs, assignments, policy)
    ctx.extra.setdefault("dirty", {})[nav.key] = nav.domain
    ctx.extra.pop("elements", None)
    if landed is not None:
        ctx.extra["state"], ctx.extra["state_key"] = landed, nav.key
    # the values are reported to the caller (they are the user's own words) but never written to the graph
    return {"set": {a.var: a.value for _, a in chosen}, "how": how, **_page_summary(nav, landed or ""),
            "page_changed": landed is not None and landed != before,
            "next": "call observe to see the page and get fresh element ids"}


def scroll(ctx: ToolContext, args: dict):
    page = get_nav(ctx).session.page
    where = args["direction"]
    page.evaluate({"down": "window.scrollBy(0, window.innerHeight * 0.8)",
                   "up": "window.scrollBy(0, -window.innerHeight * 0.8)",
                   "top": "window.scrollTo(0, 0)",
                   "bottom": "window.scrollTo(0, document.documentElement.scrollHeight)"}[where])
    rehearse(ctx, "scroll", f"scroll {where}", None, value=where)
    return {"scrolled": where, "scrollY": page.evaluate("window.scrollY")}


def read_text(ctx: ToolContext, args: dict):
    nav = get_nav(ctx)
    limit = int(args.get("max_chars", 2000))
    if args.get("element_id"):
        text = _resolve(ctx, nav, _element(ctx, args["element_id"]), None).locator.inner_text()
    else:
        text = nav.session.page.inner_text("body")
    return {"text": " ".join(text.split())[:limit]}


def open_url(ctx: ToolContext, args: dict):
    nav = get_nav(ctx)
    url = args["url"]
    if domain_of(url) != ctx.domain:
        ctx.require_approval(f"{url} leaves the explored site ({ctx.domain})")
    if policy_of(ctx).respect_robots and not robots_allowed(url):
        raise ValueError("robots.txt does not allow opening this URL")
    nav.session.goto(url)
    if domain_of(url) == ctx.domain:                           # leaving the site needed an approval: never replayed
        rehearse(ctx, "goto", f"open {url}", None, url=url)
    ctx.extra.pop("elements", None)
    nav = get_nav(ctx)                       # now on the new site, if it is another one
    ctx.extra["state"] = _write(ctx, nav, nav.observe())
    return {**_page_summary(nav, ctx.extra["state"]), "next": "call observe to see the page"}


def _el_schema(**extra) -> dict:
    return {"type": "object", "properties": {"element_id": {"type": "string"}, **extra}, "required": ["element_id"]}


def register(box: Toolbox) -> None:
    box.register(ToolSpec(
        "observe", "Look at the current page: lists its interactive elements with ids (e1, e2, ...), roles, names, "
        "risk class, and which are repeated lists or hover menus. Opens the site's entry page first if no page is open. "
        "Call this before click/fill/hover and again after anything changes the page.",
        {"type": "object", "properties": {}}, observe, "read", untrusted_output=True))
    box.register(ToolSpec(
        "list_items", "For an element that is a repeated list (observe shows repeated_items), read every item's text "
        "from the live page, so you can choose one by name for click.", _el_schema(), list_items, "read",
        untrusted_output=True))
    box.register(ToolSpec(
        "click", "Click an element by id. For a repeated list also pass `item` (the text of the entry to click, from "
        "list_items). Elements above the allowed risk level (checkout, delete, post, external links) need the user's "
        "approval; if the result says needs_approval, ask the user and do not retry until they approve.",
        _el_schema(item={"type": "string"}), click, "act", untrusted_output=True))
    box.register(ToolSpec(
        "hover", "Hover over an element (desktop menus that open on hover). Then call observe.",
        _el_schema(item={"type": "string"}), hover, "act", untrusted_output=True))
    box.register(ToolSpec(
        "fill", "Type a value into a text or search box. Never use for passwords. Pass the user's own words as "
        "`value`; it is typed but never stored. To type a stored secret (see list_secrets) pass `from_secret` instead: "
        "you never see its value. Optionally name the variable (var_name), e.g. 'query'.",
        _el_schema(value={"type": "string"}, from_secret={"type": "string"}, var_name={"type": "string"})
        | {"required": ["element_id"]},
        fill, "act", sensitive_args=("value",)))
    box.register(ToolSpec(
        "select_option", "Choose an option of a dropdown (observe lists its `options`), e.g. a sort order or a "
        "category filter. Pass the option's label exactly as listed.",
        _el_schema(option={"type": "string"}) | {"required": ["element_id", "option"]}, select_option, "act",
        untrusted_output=True))
    box.register(ToolSpec(
        "set_checked", "Check or uncheck a checkbox, radio button or switch (observe shows `checked`), e.g. a "
        "filter such as 'In stock'.",
        _el_schema(checked={"type": "boolean"}) | {"required": ["element_id", "checked"]}, set_checked, "act",
        untrusted_output=True))
    box.register(ToolSpec(
        "press", "Press a key: " + ", ".join(PRESS_KEYS) + ". With element_id the key goes to that element (Enter "
        "in a search box without a button, Escape in a dialog); Enter in a form counts as submitting it and may need "
        "approval.",
        {"type": "object", "properties": {"key": {"type": "string", "enum": list(PRESS_KEYS)},
                                          "element_id": {"type": "string"}}, "required": ["key"]},
        press, "act", untrusted_output=True))
    box.register(ToolSpec(
        "fill_inputs", "Use the page's own search and filter controls for a goal: fills the search box, picks "
        "dropdown / radio options, ticks checkboxes and sets numbers or dates the goal implies (never passwords, "
        "e-mail, phone, payment, address or name fields), then submits. Faster than clicking around for 'find X'.",
        {"type": "object", "properties": {"goal": {"type": "string"}}, "required": ["goal"]}, fill_inputs, "act",
        untrusted_output=True, sensitive_args=("goal",)))
    box.register(ToolSpec(
        "scroll", "Scroll the page.", {"type": "object", "properties": {"direction": {
            "type": "string", "enum": ["up", "down", "top", "bottom"]}}, "required": ["direction"]}, scroll, "act"))
    box.register(ToolSpec(
        "read_text", "Read the visible text of an element, or of the whole page when no element_id is given.",
        {"type": "object", "properties": {"element_id": {"type": "string"}, "max_chars": {"type": "integer",
                                                                                         "maximum": 8000}}},
        read_text, "read", untrusted_output=True))
    box.register(ToolSpec(
        "open_url", "Open a URL. Staying on the explored site is free; any other site needs the user's approval.",
        {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}, open_url, "act",
        untrusted_output=True))
