"""Housekeeping: delete blobs that no graph refers to any more."""
from __future__ import annotations

import time

REF_PROPS = ["screenshot_ref", "snapshot_ref", "video_ref", "trace_ref", "before_ref", "after_ref", "blob_ref",
             "blob_md", "blob_html"]


def referenced_blobs(store, keys) -> set[str]:
    refs: set[str] = set()
    for k in keys:
        for p in REF_PROPS:
            for (v,) in store.ro_query(k, f"MATCH (n) WHERE n.{p} IS NOT NULL RETURN n.{p}"):
                refs.add(v)
    return refs


def blob_gc(store, blobs, keys, grace_s: int = 3600) -> int:
    """Remove blob files that no node in `keys` references and that are older than the grace period (a blob being
    written right now is not yet referenced). Returns how many were removed."""
    refs = referenced_blobs(store, keys)
    cutoff = time.time() - grace_s
    removed = 0
    for p in blobs.root.rglob("*"):
        if p.is_file() and p.name not in refs and p.stat().st_mtime < cutoff:
            p.unlink()
            removed += 1
    return removed
