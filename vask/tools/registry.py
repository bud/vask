"""Tool registration and execution."""

from __future__ import annotations

from typing import Any

from vask.core.types import Tool, ToolResult
from vask.tools.base import ToolProvider


class ToolRegistry:
    """Manages available tools and dispatches execution."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolProvider] = {}

    def register(self, tool: ToolProvider) -> None:
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get_tool_definitions(self) -> list[Tool]:
        return [
            Tool(name=t.name, description=t.description, parameters=t.parameters)
            for t in self._tools.values()
        ]

    async def execute(self, name: str, params: dict[str, Any], tool_call_id: str) -> ToolResult:
        if name not in self._tools:
            return ToolResult(
                tool_call_id=tool_call_id,
                output=f"Unknown tool: {name}",
                is_error=True,
            )
        try:
            output = await self._tools[name].execute(params)
            return ToolResult(tool_call_id=tool_call_id, output=output)
        except Exception as e:
            return ToolResult(tool_call_id=tool_call_id, output=str(e), is_error=True)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())
