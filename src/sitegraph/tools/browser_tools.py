"""Tools that look at and act on the live page. All browser access goes through here, so the risk gate cannot be
bypassed: an element whose risk class is above the active policy needs the user's one-time approval."""
from __future__ import annotations

import re
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

from ..browser.items import choose_item, live_item_texts
from ..graph.queries import entry_state
from ..normalize import domain_of, var_name
from ..policy import robots_allowed
from .core import ToolContext, ToolSpec, Toolbox
from .graph_tools import policy_of

MAX_ELEMENTS = 150


def get_nav(ctx: ToolContext):
    """The context's Navigator (its resolver, writer and settle helpers), on the context's single browser."""
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
    return nav.write_state(scrub_typed(obs, ctx.extra.get("typed", [])))


def _ensure_page(ctx: ToolContext, nav) -> None:
    if nav.session.page.url in ("about:blank", ""):
        entry = entry_state(ctx.store, ctx.key)
        if entry:
            nav.session.goto(entry[1])


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
                listing.append(item)
    ctx.extra["elements"], ctx.extra["state"] = ids, state
    return {**_page_summary(nav, state), "elements": listing, "truncated": len(ids) > MAX_ELEMENTS}


def list_items(ctx: ToolContext, args: dict):
    el = _element(ctx, args["element_id"])
    if not el.is_template:
        raise ValueError("this element is not a repeated list item")
    return {"items": live_item_texts(get_nav(ctx).session.page, el.props())}


def _after(ctx: ToolContext, nav, before: str | None, el, kind: str) -> dict:
    obs = nav.observe()
    after = _write(ctx, nav, obs)
    filled = ctx.extra.pop("filled", {})
    if before and after != before:
        nav.writer.write_action(before, after, el.props(), kind, ",".join(filled.values()) or None, (), list(filled))
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


def _acted(ctx: ToolContext, kind: str, el) -> None:
    """Remember what was done on the page (the recovery agent's actions can be learned as steps of a route). Actions
    above the policy are left out: an approval covers one action and must never be replayed automatically."""
    if policy_of(ctx).allows(el):
        ctx.extra.setdefault("acted", []).append(f"{kind}|{el.gid}")


def click(ctx: ToolContext, args: dict):
    nav, el = get_nav(ctx), _element(ctx, args["element_id"])
    _guard(ctx, el)
    located = _resolve(ctx, nav, el, args.get("item"))
    before = ctx.extra.get("state")
    nav._click(located)
    _acted(ctx, "click", el)
    return _after(ctx, nav, before, el, "click")


def hover(ctx: ToolContext, args: dict):
    nav, el = get_nav(ctx), _element(ctx, args["element_id"])
    _guard(ctx, el)
    nav._hover(_resolve(ctx, nav, el, args.get("item")))
    _acted(ctx, "hover", el)
    return {"hovered": el.name_sample[:60], "next": "call observe to see anything that appeared (menus, tooltips)"}


def fill(ctx: ToolContext, args: dict):
    nav, el = get_nav(ctx), _element(ctx, args["element_id"])
    if el.role not in ("textbox", "searchbox", "combobox"):
        raise ValueError(f"cannot type into a {el.role}")
    if (el.input_kind or "").lower() == "password":
        raise ValueError("refusing to type into a password field")
    secret_var = args.get("from_secret")
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
    return {"filled": el.name_sample[:60], "variable": name}


def scroll(ctx: ToolContext, args: dict):
    page = get_nav(ctx).session.page
    where = args["direction"]
    page.evaluate({"down": "window.scrollBy(0, window.innerHeight * 0.8)",
                   "up": "window.scrollBy(0, -window.innerHeight * 0.8)",
                   "top": "window.scrollTo(0, 0)",
                   "bottom": "window.scrollTo(0, document.documentElement.scrollHeight)"}[where])
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
    ctx.extra.pop("elements", None)
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
