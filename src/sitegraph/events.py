from __future__ import annotations

import json


class Events:
    def __init__(self, redis_client, prefix: str = "events"):
        self.r, self.prefix = redis_client, prefix

    def channel(self, uid: str) -> str:
        return f"{self.prefix}:{uid}"

    def publish(self, uid: str, event: dict) -> None:
        self.r.publish(self.channel(uid), json.dumps(event))


class NullEvents:
    def channel(self, uid: str) -> str:
        return f"events:{uid}"

    def publish(self, uid: str, event: dict) -> None:
        pass
