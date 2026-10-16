import hashlib
import os
import time


def now_ms() -> int:
    return int(time.time() * 1000)


def gid(prefix: str, *parts: object) -> str:
    raw = "\x1f".join("" if p is None else str(p) for p in parts)
    return f"{prefix}:{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def ulid(prefix: str) -> str:
    return f"{prefix}:{now_ms():012x}{os.urandom(6).hex()}"
