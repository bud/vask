"""Tests for vask.tools.registry.ToolRegistry."""

import pytest

from vask.core.types import Tool
from vask.tools.registry import ToolRegistry


class TestToolRegistry:
    def test_register_and_list_tools(self, mock_tool):
        reg = ToolRegistry()
        reg.register(mock_tool)
        assert "mock_tool" in reg.list_tools()

    def test_list_tools_empty(self):
        reg = ToolRegistry()
        assert reg.list_tools() == []

    @pytest.mark.asyncio
    async def test_execute_known_tool_returns_result(self, mock_tool):
        reg = ToolRegistry()
        reg.register(mock_tool)
        result = await reg.execute("mock_tool", {"input": "test"}, "tc1")
        assert result.output == "mock result"
        assert result.tool_call_id == "tc1"
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_execute_unknown_tool_returns_error(self):
        reg = ToolRegistry()
        result = await reg.execute("nonexistent", {}, "tc2")
        assert result.is_error is True
        assert "Unknown tool" in result.output

    @pytest.mark.asyncio
    async def test_execute_tool_that_raises_returns_error_result(self, mock_tool):
        mock_tool.execute.side_effect = RuntimeError("boom")
        reg = ToolRegistry()
        reg.register(mock_tool)
        result = await reg.execute("mock_tool", {}, "tc3")
        assert result.is_error is True
        assert "boom" in result.output

    def test_get_tool_definitions_returns_correct_tool_objects(self, mock_tool):
        reg = ToolRegistry()
        reg.register(mock_tool)
        defs = reg.get_tool_definitions()
        assert len(defs) == 1
        assert isinstance(defs[0], Tool)
        assert defs[0].name == "mock_tool"
        assert defs[0].description == "A mock tool for testing"
        assert defs[0].parameters == mock_tool.parameters

    def test_unregister_tool(self, mock_tool):
        reg = ToolRegistry()
        reg.register(mock_tool)
        assert "mock_tool" in reg.list_tools()
        reg.unregister("mock_tool")
        assert "mock_tool" not in reg.list_tools()
