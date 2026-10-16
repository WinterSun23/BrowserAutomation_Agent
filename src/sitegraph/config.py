from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class Settings:
    uid: str = "local"
    falkor_host: str = "localhost"
    falkor_port: int = 6379
    data_dir: Path = Path("data")
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    llm_model: str = "openai/gpt-4o-mini"
    llm_api_base: str | None = None
    llm_api_key: str | None = None
    retention_runs: int = 100
    near_dup_threshold: float = 0.85
    cost_lambda: float = 2.0
    max_replans: int = 3
    template_min_items: int = 4
    min_target_score: float = 0.35
    sample_input_text: str = "test"
    headless: bool = True

    @property
    def blob_dir(self) -> Path:
        return Path(self.data_dir) / "blobs"


_ENV = {
    "falkor_host": "SITEGRAPH_FALKOR_HOST",
    "falkor_port": "SITEGRAPH_FALKOR_PORT",
    "uid": "SITEGRAPH_UID",
    "headless": "SITEGRAPH_HEADLESS",
    "llm_model": "LLM_MODEL",
    "llm_api_base": "LLM_API_BASE",
    "llm_api_key": "LLM_API_KEY",
}


def _coerce(value, current):
    if isinstance(current, bool):
        return str(value).lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, Path):
        return Path(value)
    return value


def load_settings(path: str | Path | None = "config.yaml") -> Settings:
    s = Settings()
    if path and Path(path).exists():
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        for f in fields(Settings):
            if f.name in data:
                setattr(s, f.name, _coerce(data[f.name], getattr(s, f.name)))
    for name, env in _ENV.items():
        if env in os.environ:
            setattr(s, name, _coerce(os.environ[env], getattr(s, name)))
    return s
