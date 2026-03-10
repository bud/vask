"""Config-driven LLM provider selection."""

from __future__ import annotations

from vask.config import VaskConfig
from vask.core.registry import registry
from vask.llm.base import LLMProvider


def get_llm(config: VaskConfig, name: str | None = None) -> LLMProvider:
    """Get an LLM provider by name or use the default."""
    name = name or config.defaults.get("llm", "gemini")
    provider_config = config.get_provider(name)
    return registry.create("llm", name, provider_config)
