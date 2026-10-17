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


def test_dotenv_file_is_read_and_real_env_wins(tmp_path, monkeypatch):
    import os

    p = tmp_path / "c.yaml"
    p.write_text("llm_model: openai/from-yaml\n")
    env = tmp_path / ".env"
    env.write_text("LLM_MODEL=ollama_chat/from-dotenv\nLLM_API_BASE=http://localhost:11434\n")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_API_BASE", raising=False)
    s = load_settings(p, env_file=env)
    assert s.llm_model == "ollama_chat/from-dotenv" and s.llm_api_base == "http://localhost:11434"
    assert "LLM_MODEL" not in os.environ                      # read into settings only, never leaked to the process
    monkeypatch.setenv("LLM_MODEL", "openai/from-env")
    assert load_settings(p, env_file=env).llm_model == "openai/from-env"
    monkeypatch.delenv("LLM_MODEL")
    assert load_settings(p, env_file=tmp_path / "missing.env").llm_model == "openai/from-yaml"
    monkeypatch.setenv("SITEGRAPH_ENV_FILE", str(env))                  # the default location can be redirected
    assert load_settings(p).llm_model == "ollama_chat/from-dotenv"
    monkeypatch.setenv("SITEGRAPH_ENV_FILE", "")                        # and switched off (what the test suite does)
    assert load_settings(p).llm_model == "openai/from-yaml"
