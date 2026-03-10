"""Simple provider registry — maps names to factory functions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vask.config import ProviderConfig


class Registry:
    """Stores factory functions for inputs, transcribers, LLMs, outputs, and tools."""

    def __init__(self) -> None:
        self._factories: dict[str, dict[str, Callable[..., Any]]] = {
            "input": {},
            "transcription": {},
            "llm": {},
            "output": {},
            "tool": {},
        }

    def register(self, category: str, name: str, factory: Callable[..., Any]) -> None:
        if category not in self._factories:
            raise ValueError(f"Unknown category '{category}'. Must be one of {list(self._factories)}")
        self._factories[category][name] = factory

    def create(self, category: str, name: str, config: ProviderConfig | None = None) -> Any:
        if name not in self._factories.get(category, {}):
            available = list(self._factories.get(category, {}).keys())
            raise KeyError(f"No '{category}' provider named '{name}'. Available: {available}")
        factory = self._factories[category][name]
        if config is not None:
            return factory(config)
        return factory()

    def list_providers(self, category: str) -> list[str]:
        return list(self._factories.get(category, {}).keys())


# Global registry instance
registry = Registry()


def register_defaults() -> None:
    """Register all built-in providers."""
    from vask.inputs.clipboard import ClipboardInput
    from vask.inputs.file import FileInput
    from vask.inputs.mic import MicInput
    from vask.llm.anthropic import AnthropicLLM
    from vask.llm.gemini import GeminiLLM
    from vask.llm.openai import OpenAILLM
    from vask.outputs.clipboard import ClipboardOutput
    from vask.outputs.json_out import JsonOutput
    from vask.outputs.terminal import TerminalOutput
    from vask.tools.http import HttpTool
    from vask.tools.shell import ShellTool
    from vask.transcribe.openai import OpenAIWhisper

    # Inputs
    registry.register("input", "mic", MicInput)
    registry.register("input", "file", FileInput)
    registry.register("input", "clipboard", ClipboardInput)

    # Transcription
    registry.register("transcription", "openai-whisper", OpenAIWhisper)

    # LLMs
    registry.register("llm", "gemini", GeminiLLM)
    registry.register("llm", "claude", AnthropicLLM)
    registry.register("llm", "openai", OpenAILLM)
    registry.register("llm", "openrouter", OpenAILLM)  # OpenRouter uses OpenAI-compatible API

    # Outputs
    registry.register("output", "terminal", TerminalOutput)
    registry.register("output", "clipboard", ClipboardOutput)
    registry.register("output", "json", JsonOutput)

    # Tools
    registry.register("tool", "http", HttpTool)
    registry.register("tool", "shell", ShellTool)
