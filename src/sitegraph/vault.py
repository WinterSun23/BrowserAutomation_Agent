from __future__ import annotations

import os

_MEMORY: dict[str, str] = {}


class Vault:
    """Secret values live outside every graph (spec §10). OS keyring by default; SITEGRAPH_VAULT=memory for tests."""

    SERVICE = "sitegraph"

    def __init__(self, uid: str):
        self.uid = uid
        self.memory = os.environ.get("SITEGRAPH_VAULT") == "memory"

    def set(self, vault_key: str, value: str) -> None:
        if self.memory:
            _MEMORY[vault_key] = value
            return
        import keyring

        keyring.set_password(self.SERVICE, vault_key, value)

    def get(self, vault_key: str) -> str | None:
        if self.memory:
            return _MEMORY.get(vault_key)
        import keyring

        return keyring.get_password(self.SERVICE, vault_key)

    def delete(self, vault_key: str) -> None:
        if self.memory:
            _MEMORY.pop(vault_key, None)
            return
        import keyring

        try:
            keyring.delete_password(self.SERVICE, vault_key)
        except Exception:
            pass
