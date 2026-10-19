from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml
from dotenv import dotenv_values


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
    policy: dict = field(default_factory=dict)  # per-run overrides of the exploration policy (see ExplorationPolicy)
    visual_diff_bits: int = 6          # dHash bits that must differ before a same-state click counts as a visual change
    use_git: bool = True
    checkpoint_every: int = 20
    compact_keep_commits: int = 200
    drift_heal_rate: float = 0.3
    drift_min_samples: int = 3
    quarantine_floor: float = 0.25
    quarantine_min_seen: int = 4
    uipattern_sim: float = 0.85
    logical_state_min: float = 0.6
    workflow_similar: float = 0.8
    intent_dedupe: float = 0.95
    publish_min_seen: int = 2
    navigators_per_site: int = 2
    claim_ttl_ms: int = 120000
    claim_wait_s: float = 5.0
    claim_retries: int = 3
    state_identity: str = "coarse"     # coarse: content regions count roles, not names (Task 4); fine: v2 identity
    exploration_mode: str = "auto"     # auto | brute | guided (Tasks 9-10)
    auto_sitemap_large: int = 300      # auto: a sitemap with more URLs than this means guided
    auto_probe_states: int = 8         # auto: states explored brute-force before deciding
    auto_frontier_large: int = 15      # auto: more open frontier groups than this after the probe means guided
    guided_round_states: int = 10      # guided: new states per round
    guided_max_groups: int = 40        # guided: frontier groups shown to the model per round
    guided_max_rounds: int = 12        # guided: rounds per exploration
    plan_max_steps: int = 6            # planner: steps per request (Task 12)
    choose_margin: float = 0.05        # targets this close in score are chosen by the model (Task 12)
    goal_found_score: float = 0.6      # goal search: a target this strong counts as found on its own (Task 11)
    goal_words_min: float = 0.6        # goal search: share of goal words on the page for a weaker target
    goal_seed_best: int = 5            # goal search: best new screens that seed the next round
    goal_explore: bool = True          # a request for a screen the map lacks explores toward it first
    goal_explore_round_states: int = 12  # new screens per exploration round while searching for the goal
    goal_explore_max_rounds: int = 8   # rounds before giving up (the policy's state and time budgets also apply)
    dynamic_spawn: bool = True         # start explorers as work piles up (False: start all navigators at once)
    spawn_min_backlog: int = 4         # unclaimed open work items that must be waiting ...
    spawn_after_s: float = 3.0         # ... for this long before one more explorer is started
    spawn_poll_s: float = 0.5          # how often the backlog is checked
    max_scrolls: int = 8
    agent_max_steps: int = 60          # model turns per request (each may call tools)
    agent_max_tokens: int = 400000     # token budget per request
    agent_repeat_limit: int = 5        # the same tool call with the same result this many times stops the request (0: off)
    agent_keep_observations: int = 2   # page observations re-sent in full to the model (older ones become a stub; 0: all)
    chat_learn: bool = True            # curate the pages the chat agent visited by hand when a request ends
    demo_max_seconds: int = 120        # target length of a demo video
    demo_retakes: int = 2              # final takes the Demo agent may re-record after the first one
    demo_approve_script: bool = True   # show the demo script to the user and wait for approval before recording
    escalate_navigator: bool = True    # a stuck Navigator step may hand over to an LLM recovery agent (needs a chat LLM)
    max_escalations_per_run: int = 5   # recovery attempts allowed per execution run
    escalation_max_steps: int = 30     # model turns one recovery attempt may use
    escalate_curator: bool = True      # anomalies found after curation may be reviewed by an LLM curator agent
    curator_max_changes: int = 20      # graph edits one curator review may make
    anomaly_dup_low: float = 0.6       # states of one page this similar (but below near_dup_threshold) are flagged
    anomaly_states_per_page: int = 8   # more states than this on one page type is flagged as an explosion
    anomaly_edge_min_seen: int = 5     # an action tried at least this often ...
    anomaly_edge_reliability: float = 0.4  # ... with reliability below this is flagged
    tool_result_max_chars: int = 6000  # tool results shown to an LLM are trimmed to this length
    item_match_threshold: float = 0.6  # min cosine between request and a live list item to pick it by meaning
    item_match_margin: float = 0.05    # the best item must beat the runner-up by this much
    max_path_len: int = 2  # how many chained clicks/hovers may lead to an element (e.g. open menu -> open submenu -> item)

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


def load_settings(path: str | Path | None = "config.yaml", env_file: str | Path | None = None) -> Settings:
    """Defaults < config.yaml < .env < real environment variables. The .env file is read into the settings only;
    it never changes os.environ. `env_file` defaults to $SITEGRAPH_ENV_FILE, else '.env'; an empty value disables it."""
    if env_file is None:
        env_file = os.environ.get("SITEGRAPH_ENV_FILE", ".env")
    s = Settings()
    if path and Path(path).exists():
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        for f in fields(Settings):
            if f.name in data:
                setattr(s, f.name, _coerce(data[f.name], getattr(s, f.name)))
    env_values = dotenv_values(env_file) if env_file and Path(env_file).exists() else {}
    env_values = {**{k: v for k, v in env_values.items() if v is not None}, **os.environ}
    for name, env in _ENV.items():
        if env in env_values:
            setattr(s, name, _coerce(env_values[env], getattr(s, name)))
    return s
