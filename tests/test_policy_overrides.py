import pytest
from sitegraph.cli import apply_policy_flags, build_parser
from sitegraph.config import load_settings
from sitegraph.embed import HashEmbedder
from sitegraph.policy import ExplorationPolicy
from sitegraph.profile import Profile


def test_with_overrides_changes_only_the_named_fields():
    base = ExplorationPolicy(max_depth=3, budget_s=300)
    p = base.with_overrides({"max_depth": "6", "allow_forms": True, "deny_patterns": ["logout"]})
    assert (p.max_depth, p.allow_forms, p.deny_patterns, p.budget_s) == (6, True, ["logout"], 300)
    assert base.max_depth == 3                                       # the original is untouched
    assert base.with_overrides({}) == base


@pytest.mark.parametrize("bad", [{"max_dept": 5}, {"max_risk": "nuclear"}, {"max_depth": "deep"}, {"max_depth": -1},
                                 {"deny_patterns": "logout"}, {"deny_patterns": ["("]}])
def test_with_overrides_rejects_bad_values_with_a_clear_message(bad):
    with pytest.raises(ValueError) as e:
        ExplorationPolicy().with_overrides(bad)
    assert next(iter(bad)) in str(e.value)


def test_config_yaml_policy_block_is_loaded(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("policy:\n  max_depth: 6\n  budget_s: 120\n")
    assert load_settings(p, env_file="").policy == {"max_depth": 6, "budget_s": 120}
    assert load_settings(tmp_path / "missing.yaml", env_file="").policy == {}


def test_flags_beat_the_config_block_which_beats_the_saved_policies(store, settings):
    prof = Profile(store, settings.uid, HashEmbedder())
    prof.save_policy(ExplorationPolicy(name="default", max_depth=3, max_states=60, budget_s=300), is_default=True)
    prof.register_site("shop.test", "desktop", "k", "ok", "http://shop.test/")
    prof.save_policy(ExplorationPolicy(name="shop", max_depth=4, max_states=10), domain="shop.test", device="desktop")
    assert prof.policy_for("other.test").max_depth == 3
    assert prof.policy_for("shop.test").max_states == 10
    prof.policy_overrides = dict(settings.policy, **{"max_depth": 6, "budget_s": 120})     # config.yaml block
    assert prof.policy_for("other.test").max_depth == 6 and prof.policy_for("other.test").budget_s == 120
    assert prof.policy_for("shop.test").max_depth == 6 and prof.policy_for("shop.test").max_states == 10
    args = build_parser().parse_args(["run", "http://x.test/", "explore", "--max-depth", "9", "--budget", "30"])
    apply_policy_flags(prof, args)                                     # command-line flags win
    assert prof.policy_for("shop.test").max_depth == 9 and prof.policy_for("shop.test").budget_s == 30
    assert prof._stored_default_policy().max_depth == 3                         # nothing was written back to the graph


def test_parser_has_the_policy_flags_on_run_and_chat():
    for cmd in ("run", "chat"):
        argv = [cmd, "http://x.test/"] + (["explore"] if cmd == "run" else []) + [
            "--max-depth", "7", "--max-states", "80", "--max-actions", "200", "--budget", "60"]
        a = build_parser().parse_args(argv)
        assert (a.max_depth, a.max_states, a.max_actions, a.budget) == (7, 80, 200, 60)
    a = build_parser().parse_args(["run", "http://x.test/", "explore"])
    assert a.max_depth is None and a.budget is None
