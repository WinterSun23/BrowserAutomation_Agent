from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass

from ..ids import gid
from ..normalize import normalize_text, sample_text, url_template
from ..risk import RISK_COST, classify_risk
from .model import ComponentObs, ElementObs, RawNode, RawSnapshot, StateObs

COMPONENT_KINDS = {"navigation", "banner", "contentinfo", "main", "form", "search", "complementary", "dialog",
                   "alertdialog", "menu", "menubar", "listbox", "tablist", "tabpanel", "toolbar", "tooltip",
                   "status", "alert", "region", "list", "table", "grid", "feed", "group"}
EPHEMERAL_KINDS = {"menu", "listbox", "tooltip", "tabpanel", "tablist", "status", "alert", "group"}
ITEM_ROLES = {"listitem", "row", "article"}
CLICKABLE_ROLES = {"link", "button", "menuitem", "menuitemcheckbox", "menuitemradio", "tab", "switch"}
_ROLE_LOCATABLE = CLICKABLE_ROLES | {"textbox", "searchbox", "combobox", "checkbox", "radio", "option"}
_INPUT_KIND = {"combobox": "select", "listbox": "select", "checkbox": "checkbox", "radio": "radio"}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def absent_cs_gid(comp_gid: str) -> str:
    return gid("cs", comp_gid, "absent")


def _sha16(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


@dataclass
class _Template:
    items: list[int]
    positions: list[list[int]]


def _is_component(n: RawNode) -> bool:
    return n.role in COMPONENT_KINDS and not (n.interactive and n.tag in ("select", "input"))


def _find_templates(nodes: list[RawNode], min_items: int) -> list[_Template]:
    children: dict[int, list[int]] = defaultdict(list)
    for n in nodes:
        if n.parent is not None:
            children[n.parent].append(n.idx)

    def desc(i: int) -> list[int]:
        out: list[int] = []
        for c in children.get(i, []):
            if nodes[c].interactive:
                out.append(c)
            out.extend(desc(c))
        return out

    groups: list[_Template] = []
    for kids in list(children.values()):
        items = [k for k in kids if nodes[k].role in ITEM_ROLES]
        if len(items) < min_items:
            continue
        by_sig: dict[tuple, list[tuple[int, list[int]]]] = defaultdict(list)
        for it in items:
            d = desc(it)
            sig = tuple((nodes[x].role, nodes[x].depth - nodes[it].depth) for x in d)
            if sig:
                by_sig[sig].append((it, d))
        for members in by_sig.values():
            if len(members) < min_items:
                continue
            positions = [list(p) for p in zip(*[d for _, d in members])]
            if all(_destinations_agree(nodes, pos) for pos in positions):
                groups.append(_Template(items=[it for it, _ in members], positions=positions))
    return groups


def _destinations_agree(nodes: list[RawNode], pos: list[int]) -> bool:
    links = [nodes[i] for i in pos if nodes[i].role == "link"]
    if len(links) == len(pos):
        return len({url_template(n.href) if n.href else "" for n in links}) == 1
    return True


def _hoverable(n: RawNode) -> bool:
    s = n.states or {}
    return s.get("haspopup") not in (None, "", "false") or s.get("popup") == "true"


def _identity_key(n: RawNode) -> str:
    if n.role == "link" and n.href:
        return "href:" + url_template(n.href)
    return normalize_text(n.name)


def _component_gid(kind: str, root: RawNode | None, tpl: str) -> str:
    if root is None or kind == "main":
        return gid("cp", kind, tpl)
    return gid("cp", kind, normalize_text(root.name), root.gcss)


def build_state(raw: RawSnapshot, domain: str, template_min: int = 4) -> StateObs:
    nodes = raw.nodes
    tpl = url_template(raw.url)
    page_gid = gid("pg", domain, tpl)
    is_comp = [_is_component(n) for n in nodes]

    def comp_of(n: RawNode) -> int | None:
        p = n.parent
        while p is not None:
            if is_comp[p]:
                return p
            p = nodes[p].parent
        return None

    def form_of(n: RawNode) -> int | None:
        p = n.parent
        while p is not None:
            if nodes[p].role in ("form", "search"):
                return p
            p = nodes[p].parent
        return None

    templates = _find_templates(nodes, template_min)
    tpl_rep: dict[int, tuple[_Template, int]] = {}
    tpl_member: set[int] = set()
    for t in templates:
        for j, pos in enumerate(t.positions):
            tpl_rep[pos[0]] = (t, j)
            tpl_member.update(pos)

    members: dict[int | None, list[RawNode]] = defaultdict(list)
    for n in nodes:
        if n.interactive and not is_comp[n.idx]:
            members[comp_of(n)].append(n)

    comp_gids: dict[int | None, str] = {}
    for idx, n in enumerate(nodes):
        if is_comp[idx]:
            comp_gids[idx] = _component_gid(n.role, n, tpl)
    comp_gids[None] = _component_gid("page", None, tpl)

    components: list[ComponentObs] = []
    for root_idx, elems in members.items():
        root = nodes[root_idx] if root_idx is not None else None
        kind = root.role if root else "page"
        comp_gid = comp_gids[root_idx]
        blocking = kind in ("dialog", "alertdialog") and root is not None and root.states.get("modal") == "true"
        ephemeral = kind in EPHEMERAL_KINDS
        built: list[tuple[RawNode, str, dict]] = []  # node, key, extra
        for n in elems:
            if n.idx in tpl_member and n.idx not in tpl_rep:
                continue
            built.append((n, _identity_key(n), {}))
        tokens = []
        seen: dict[tuple[str, str], int] = defaultdict(int)
        elements: list[ElementObs] = []
        aria_state: dict = {}
        for n, key, _ in built:
            if n.idx in tpl_rep:
                t, j = tpl_rep[n.idx]
                pos = t.positions[j]
                raw_names = {sample_text(nodes[i].name) for i in pos}
                key = key if key.startswith("href:") else (
                    normalize_text(nodes[pos[0]].name) if len(raw_names) == 1 else "<item>")
                tokens.append(f"{n.role}|{key}|T")
            else:
                tokens.append(f"{n.role}|{key}")
        skeleton_hash = _sha16("\n".join(tokens))
        cs_gid = gid("cs", comp_gid, skeleton_hash)
        for n, key, _ in built:
            name_norm = normalize_text(n.name)
            fp = {"role": n.role, "name": n.name, "tag": n.tag, "attrs": n.attrs, "gcss": n.gcss, "css": n.css,
                  "bbox": n.bbox}
            risk = classify_risk(n.role, name_norm, n.href, domain)
            form_idx = form_of(n)
            common = dict(cs_gid=cs_gid, role=n.role, name_sample=sample_text(n.name), tag=n.tag, fingerprint=fp,
                          risk=risk, risk_cost=RISK_COST[risk], css=n.css,
                          input_kind=(n.input_type or "text") if n.role in ("textbox", "searchbox")
                          else _INPUT_KIND.get(n.role),
                          href_template=url_template(n.href) if n.href else None,
                          in_form=comp_gids.get(form_idx) if form_idx is not None else None,
                          hoverable=_hoverable(n), inferred=n.inferred)
            if n.idx in tpl_rep:
                t, j = tpl_rep[n.idx]
                pos = t.positions[j]
                item = nodes[t.items[0]]
                constant = len({sample_text(nodes[i].name) for i in pos}) == 1
                tname = normalize_text(n.name) if constant else "<item>"
                rel = n.gcss[len(item.gcss) + 3:] if n.gcss.startswith(item.gcss + " > ") else n.tag
                key_fields = [nodes[p[0]].role for p in t.positions
                              if len({sample_text(nodes[i].name) for i in p}) > 1]
                elements.append(ElementObs(
                    gid=gid("el", comp_gid, n.role, key if key.startswith("href:") else tname, "tpl", item.gcss),
                    name_norm=tname, locators=[f"item={item.gcss}||{rel}"], is_template=True, count=len(t.items),
                    item_locator=f"css={item.gcss}", item_key_fields=key_fields,
                    sample_items=[sample_text(nodes[i].name, 80) for i in t.items[:60]], **common))
                continue
            sib = seen[(n.role, key)]
            seen[(n.role, key)] += 1
            locs = []
            if n.name and n.role in _ROLE_LOCATABLE:
                locs.append(f"role={n.role}|name={n.name}")
            if n.testid:
                locs.append(f"testid={n.testid}")
            locs.append(f"css={n.css}")
            if n.name and n.role in ("link", "button") and len(n.name) <= 80:
                locs.append(f"text={n.name}")
            st = {k: v for k, v in n.states.items() if k in ("expanded", "selected", "checked") and v is not None}
            if st:
                aria_state[n.css] = st
            elements.append(ElementObs(gid=gid("el", comp_gid, n.role, key, sib), name_norm=name_norm,
                                       locators=locs, **common))
        if not elements and not blocking:
            continue
        cs_name = "open" if (ephemeral or blocking) else f"v{skeleton_hash[:6]}"
        components.append(ComponentObs(gid=comp_gid, kind=kind, name=sample_text(root.name if root else "", 80),
                                       blocking=blocking, ephemeral=ephemeral, root_css=root.gcss if root else "body",
                                       cs_gid=cs_gid, cs_name=cs_name, skeleton_hash=skeleton_hash,
                                       aria_state=aria_state, elements=elements,
                                       root_bbox=root.bbox if root else None))

    identity = [c for c in components if not c.ephemeral]
    state_gid = gid("st", page_gid, *sorted(c.cs_gid for c in identity))
    tokens = sorted({tpl} | {f"{c.kind}|{e.role}|{e.href_template or e.name_norm}" for c in identity for e in c.elements})
    return StateObs(url=raw.url, url_template=tpl, title=raw.title, page_gid=page_gid, state_gid=state_gid,
                    skeleton_hash=_sha16("\n".join(tokens)), tokens=tokens,
                    blocking_sig=",".join(sorted(c.gid for c in components if c.blocking)),
                    components=components, raw=raw)
