"""Protocol for tool providers."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ToolProvider(Protocol):
    """An executable tool the LLM can invoke."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema

    async def execute(self, params: dict[str, Any]) -> str: ...
