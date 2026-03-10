"""Tests for vask.core.pipeline.Pipeline."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vask.config import VaskConfig
from vask.core.pipeline import Pipeline
from vask.core.types import Message, Query, Response, Role, ToolCall, ToolResult


class TestPipelineAsk:
    @pytest.mark.asyncio
    async def test_ask_sends_query_to_llm_and_returns_response(self, mock_config, mock_llm):
        pipeline = Pipeline(mock_config)

        mock_output = AsyncMock()
        mock_output.render = AsyncMock()

        with (
            patch.object(pipeline, "_get_llm", return_value=mock_llm),
            patch.object(pipeline, "_get_output", return_value=mock_output),
            patch.object(pipeline, "_get_tool_definitions", return_value=[]),
        ):
            response = await pipeline.ask("What is 2+2?")

        assert response.text == "test response"
        mock_llm.complete.assert_awaited_once()
        # Verify the user message was added to history
        assert any(m.content == "What is 2+2?" for m in pipeline.ctx.history)


class TestPipelineToolLoop:
    @pytest.mark.asyncio
    async def test_tool_loop_executes_tools_and_feeds_results_back(self, mock_config):
        pipeline = Pipeline(mock_config)

        # First call returns a tool call, second call returns final response
        tc = ToolCall(id="tc1", name="mock_tool", arguments={"input": "test"})
        response_with_tool = Response(text="", tool_calls=[tc])
        final_response = Response(text="final answer")

        mock_llm = AsyncMock()
        # _tool_loop receives response_with_tool as initial response, executes the
        # tool call, then calls complete_with_tools once which should return final.
        mock_llm.complete_with_tools = AsyncMock(return_value=final_response)

        query = Query(messages=[Message(role=Role.USER, content="test")])

        # Mock tool execution
        pipeline.tool_registry.execute = AsyncMock(
            return_value=ToolResult(tool_call_id="tc1", output="tool output")
        )

        from vask.core.types import Tool

        tools = [Tool(name="mock_tool", description="test", parameters={})]
        result = await pipeline._tool_loop(mock_llm, query, tools, response_with_tool)

        assert result.text == "final answer"
        pipeline.tool_registry.execute.assert_awaited_once_with("mock_tool", {"input": "test"}, "tc1")
        # Verify tool result was appended to query messages
        assert any(m.role == Role.TOOL for m in query.messages)

    @pytest.mark.asyncio
    async def test_tool_loop_respects_max_iterations(self, mock_config):
        pipeline = Pipeline(mock_config)

        tc = ToolCall(id="tc1", name="mock_tool", arguments={})
        # LLM always returns tool calls -- should stop after max_iterations
        always_tool_response = Response(text="", tool_calls=[tc])

        mock_llm = AsyncMock()
        mock_llm.complete_with_tools = AsyncMock(return_value=always_tool_response)

        pipeline.tool_registry.execute = AsyncMock(
            return_value=ToolResult(tool_call_id="tc1", output="result")
        )

        query = Query(messages=[Message(role=Role.USER, content="test")])
        from vask.core.types import Tool

        tools = [Tool(name="mock_tool", description="test", parameters={})]

        result = await pipeline._tool_loop(
            mock_llm, query, tools, always_tool_response, max_iterations=3
        )

        # Should have been called exactly 3 times (max_iterations)
        assert mock_llm.complete_with_tools.await_count == 3
