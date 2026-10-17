"""Tools for the user's secrets and safety policy. Changing either is `risky`: a model must not do it unasked, so the
user approves each call. Secret values never come back out of these tools."""
from __future__ import annotations

from ..policy import ExplorationPolicy
from .core import ToolContext, ToolSpec, Toolbox


def _profile(ctx: ToolContext):
    if ctx.profile is None:
        raise ValueError("the user's profile is not available in this session")
    return ctx.profile


def _set_secret(ctx: ToolContext, args: dict):
    scope = args.get("scope") or ctx.domain or "*"
    _profile(ctx).set_secret(args["var"], args["value"], scope)
    return {"saved": args["var"], "scope": scope}


def _list_secrets(ctx: ToolContext, args: dict):
    return {"secrets": _profile(ctx).secret_vars(),
            "note": "names only; use fill with from_secret=<var> to type one without seeing it"}


def _set_policy(ctx: ToolContext, args: dict):
    policy = ExplorationPolicy.from_dict(args["policy"])
    site_only = bool(args.get("this_site_only"))
    domain = ctx.domain if site_only else None
    _profile(ctx).save_policy(policy, is_default=not site_only, domain=domain, device=ctx.device)
    if site_only:
        ctx.policy = policy                                 # takes effect for the rest of this conversation
    return {"saved": policy.name, "applies_to": domain or "all sites by default", "policy": policy.to_dict()}


def register(box: Toolbox) -> None:
    box.register(ToolSpec(
        "set_secret", "Store a secret (login, email, key) in the user's vault under a variable name, for one site "
        "(scope = its domain, the default is this site) or '*' for all. Only on the user's explicit request. The value is "
        "never shown again; later type it with fill(from_secret=<var>).",
        {"type": "object", "properties": {"var": {"type": "string"}, "value": {"type": "string"},
                                          "scope": {"type": "string"}}, "required": ["var", "value"]},
        _set_secret, "risky", sensitive_args=("value",)))
    box.register(ToolSpec(
        "list_secrets", "List the names (never the values) of the secrets in the vault.",
        {"type": "object", "properties": {}}, _list_secrets, "read"))
    box.register(ToolSpec(
        "set_policy", "Change the exploration/safety policy (allowed risk level, depth, limits, forms, deny patterns, "
        "robots). Only on the user's explicit request. Set this_site_only to change it for this site alone.",
        {"type": "object", "properties": {
            "policy": {"type": "object", "description": "fields of ExplorationPolicy, e.g. {\"name\": \"p\", "
                                                        "\"max_risk\": \"submit_safe\", \"allow_forms\": true}"},
            "this_site_only": {"type": "boolean"}}, "required": ["policy"]}, _set_policy, "risky"))
