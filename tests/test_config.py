from sitegraph.config import load_settings, Settings


def test_defaults_without_file(tmp_path):
    s = load_settings(tmp_path / "missing.yaml")
    assert s.embed_dim == 384 and s.retention_runs == 100


def test_yaml_and_env_override(tmp_path, monkeypatch):
    p = tmp_path / "c.yaml"
    p.write_text("retention_runs: 7\nheadless: false\n")
    monkeypatch.setenv("LLM_MODEL", "openai/my-model")
    s = load_settings(p)
    assert s.retention_runs == 7 and s.headless is False and s.llm_model == "openai/my-model"
    assert s.blob_dir == s.data_dir / "blobs"
