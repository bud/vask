"""Tests for vask.config."""

from pathlib import Path

import pytest

from vask.config import ProviderConfig, VaskConfig, load_config


class TestLoadConfig:
    def test_load_config_nonexistent_file_returns_defaults(self, tmp_path):
        config = load_config(tmp_path / "nonexistent.toml")
        assert config.defaults["input"] == "mic"
        assert config.defaults["llm"] == "gemini"
        assert config.defaults["output"] == "terminal"
        assert config.providers == {}

    def test_load_config_with_valid_toml(self, tmp_path):
        config_path = tmp_path / "config.toml"
        config_path.write_text("""\
[defaults]
llm = "claude"

[providers.claude]
type = "llm"
api_key_env = "ANTHROPIC_API_KEY"
model = "claude-sonnet-4-20250514"
""")
        config = load_config(config_path)
        assert config.defaults["llm"] == "claude"
        assert "claude" in config.providers
        assert config.providers["claude"].model == "claude-sonnet-4-20250514"


class TestVaskConfig:
    def test_get_provider_raises_key_error_for_missing(self):
        config = VaskConfig()
        with pytest.raises(KeyError, match="Provider 'nonexistent' not found"):
            config.get_provider("nonexistent")

    def test_get_provider_returns_config(self):
        pc = ProviderConfig(name="test", type="llm")
        config = VaskConfig(providers={"test": pc})
        assert config.get_provider("test") is pc


class TestProviderConfig:
    def test_api_key_reads_from_env(self, monkeypatch):
        monkeypatch.setenv("TEST_API_KEY_12345", "secret-key-value")
        pc = ProviderConfig(name="test", type="llm", api_key_env="TEST_API_KEY_12345")
        assert pc.api_key == "secret-key-value"

    def test_api_key_returns_empty_when_env_not_set(self):
        pc = ProviderConfig(name="test", type="llm", api_key_env="DEFINITELY_NOT_SET_XYZ")
        assert pc.api_key == ""

    def test_api_key_returns_empty_when_no_env_var_configured(self):
        pc = ProviderConfig(name="test", type="llm")
        assert pc.api_key == ""
