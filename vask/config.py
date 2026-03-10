"""Configuration loader — TOML config file + environment variables."""

from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 12):
    import tomllib
else:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]


CONFIG_DIR = Path(os.environ.get("VASK_CONFIG_DIR", "~/.config/vask")).expanduser()
CONFIG_FILE = CONFIG_DIR / "config.toml"

DEFAULT_CONFIG: dict[str, Any] = {
    "defaults": {
        "input": "mic",
        "transcription": "openai-whisper",
        "llm": "gemini",
        "output": "terminal",
    },
    "providers": {},
    "daemon": {"hotkey": "cmd+shift+v"},
    "mcp": {"transport": "stdio"},
}


@dataclass(slots=True)
class ProviderConfig:
    """Configuration for a single provider."""

    name: str
    type: str  # "transcription", "llm", "input", "output"
    api_key_env: str = ""
    model: str = ""
    base_url: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def api_key(self) -> str:
        if self.api_key_env:
            return os.environ.get(self.api_key_env, "")
        return ""


@dataclass(slots=True)
class VaskConfig:
    """Top-level Vask configuration."""

    defaults: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CONFIG["defaults"]))
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    daemon: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_CONFIG["daemon"]))
    mcp: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_CONFIG["mcp"]))

    def get_provider(self, name: str) -> ProviderConfig:
        if name not in self.providers:
            raise KeyError(f"Provider '{name}' not found in config. Available: {list(self.providers)}")
        return self.providers[name]

    def default_provider(self, provider_type: str) -> ProviderConfig:
        name = self.defaults.get(provider_type, "")
        if not name:
            raise KeyError(f"No default set for provider type '{provider_type}'")
        return self.get_provider(name)


def _merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: Path | None = None) -> VaskConfig:
    """Load config from TOML file, falling back to defaults."""
    raw = copy.deepcopy(DEFAULT_CONFIG)

    config_path = path or CONFIG_FILE
    if config_path.exists():
        with open(config_path, "rb") as f:
            file_data = tomllib.load(f)
        raw = _merge(raw, file_data)

    providers: dict[str, ProviderConfig] = {}
    for name, pconf in raw.get("providers", {}).items():
        pconf = dict(pconf)  # avoid mutating the parsed dict
        providers[name] = ProviderConfig(
            name=name,
            type=pconf.pop("type", ""),
            api_key_env=pconf.pop("api_key_env", ""),
            model=pconf.pop("model", ""),
            base_url=pconf.pop("base_url", ""),
            extra=pconf,
        )

    return VaskConfig(
        defaults=raw.get("defaults", {}),
        providers=providers,
        daemon=raw.get("daemon", {}),
        mcp=raw.get("mcp", {}),
    )


def ensure_config() -> VaskConfig:
    """Load config, creating default config file if it doesn't exist."""
    if not CONFIG_FILE.exists():
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(_default_config_toml())
    return load_config()


def _default_config_toml() -> str:
    return """\
# Vask configuration
# Set your API keys as environment variables, then reference them here.

[defaults]
input = "mic"
transcription = "openai-whisper"
llm = "gemini"
output = "terminal"

[providers.openai-whisper]
type = "transcription"
api_key_env = "OPENAI_API_KEY"
model = "whisper-1"

[providers.gemini]
type = "llm"
api_key_env = "GEMINI_API_KEY"
model = "gemini-2.0-flash"

[providers.claude]
type = "llm"
api_key_env = "ANTHROPIC_API_KEY"
model = "claude-sonnet-4-20250514"

[providers.openai]
type = "llm"
api_key_env = "OPENAI_API_KEY"
model = "gpt-4o"

[providers.openrouter]
type = "llm"
api_key_env = "OPENROUTER_API_KEY"
base_url = "https://openrouter.ai/api/v1"
model = "anthropic/claude-sonnet-4"

[daemon]
hotkey = "cmd+shift+v"

[mcp]
transport = "stdio"
"""
