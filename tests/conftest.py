"""Common test fixtures for the vask test suite."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from vask.config import ProviderConfig, VaskConfig
from vask.core.types import Query, Response, Tool
from vask.tools.registry import ToolRegistry


@pytest.fixture
def mock_config() -> VaskConfig:
    """VaskConfig with test defaults and a dummy provider."""
    return VaskConfig(
        defaults={
            "input": "mic",
            "transcription": "openai-whisper",
            "llm": "test-llm",
            "output": "terminal",
        },
        providers={
            "test-llm": ProviderConfig(
                name="test-llm",
                type="llm",
                api_key_env="",
                model="test-model",
            ),
        },
    )


@pytest.fixture
def tool_registry() -> ToolRegistry:
    """Empty ToolRegistry instance."""
    return ToolRegistry()


@pytest.fixture
def mock_llm() -> AsyncMock:
    """A mock LLM that returns canned responses."""
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=Response(text="test response", model="mock"))
    llm.complete_with_tools = AsyncMock(
        return_value=Response(text="test response", model="mock")
    )
    return llm


@pytest.fixture
def mock_tool() -> AsyncMock:
    """A mock ToolProvider with standard attributes."""
    tool = AsyncMock()
    tool.name = "mock_tool"
    tool.description = "A mock tool for testing"
    tool.parameters = {
        "type": "object",
        "properties": {"input": {"type": "string"}},
        "required": ["input"],
    }
    tool.execute = AsyncMock(return_value="mock result")
    return tool
