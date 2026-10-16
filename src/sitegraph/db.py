from __future__ import annotations

import redis
from falkordb import FalkorDB


def device_key(uid: str, domain: str, device: str) -> str:
    return f"u:{uid}:site:{domain}:{device}"


def ops_key(uid: str, domain: str) -> str:
    return f"u:{uid}:ops:{domain}"


def profile_key(uid: str) -> str:
    return f"u:{uid}:profile"


class GraphStore:
    def __init__(self, host: str = "localhost", port: int = 6379):
        self.db = FalkorDB(host=host, port=port)
        self.redis = redis.Redis(host=host, port=port)

    def query(self, key: str, cypher: str, params: dict | None = None) -> list[list]:
        return self.db.select_graph(key).query(cypher, params or {}).result_set

    def ro_query(self, key: str, cypher: str, params: dict | None = None) -> list[list]:
        return self.db.select_graph(key).ro_query(cypher, params or {}).result_set

    def list_graphs(self, prefix: str = "") -> list[str]:
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in self.db.list_graphs()]
        return [n for n in names if n.startswith(prefix)]

    def exists(self, key: str) -> bool:
        return key in self.list_graphs(key)

    def delete(self, key: str) -> None:
        if self.exists(key):
            self.db.select_graph(key).delete()

    def copy(self, src: str, dst: str) -> None:
        self.redis.execute_command("GRAPH.COPY", src, dst)

    def execute(self, *args):
        return self.redis.execute_command(*args)
