"""Tests for vask.core.types."""

from vask.core.types import (
    AudioChunk,
    Message,
    Query,
    Response,
    Role,
    ToolCall,
    ToolResult,
)


class TestAudioChunk:
    def test_creation_with_defaults(self):
        chunk = AudioChunk(data=b"\x00\x01")
        assert chunk.data == b"\x00\x01"
        assert chunk.sample_rate == 16000
        assert chunk.channels == 1
        assert chunk.format == "wav"

    def test_creation_with_custom_values(self):
        chunk = AudioChunk(data=b"\xff", sample_rate=44100, channels=2, format="pcm")
        assert chunk.sample_rate == 44100
        assert chunk.channels == 2
        assert chunk.format == "pcm"


class TestMessage:
    def test_user_message(self):
        msg = Message(role=Role.USER, content="hello")
        assert msg.role == Role.USER
        assert msg.content == "hello"

    def test_assistant_message(self):
        msg = Message(role=Role.ASSISTANT, content="hi back")
        assert msg.role == Role.ASSISTANT

    def test_system_message(self):
        msg = Message(role=Role.SYSTEM, content="system prompt")
        assert msg.role == Role.SYSTEM

    def test_tool_message(self):
        msg = Message(role=Role.TOOL, content="tool output")
        assert msg.role == Role.TOOL


class TestQuery:
    def test_defaults(self):
        query = Query(messages=[])
        assert query.messages == []
        assert query.system_prompt is None
        assert query.temperature == 0.7
        assert query.max_tokens == 4096

    def test_custom_values(self):
        msgs = [Message(role=Role.USER, content="test")]
        query = Query(messages=msgs, temperature=0.0, max_tokens=100)
        assert query.temperature == 0.0
        assert query.max_tokens == 100
        assert len(query.messages) == 1


class TestResponse:
    def test_has_tool_calls_false_when_empty(self):
        resp = Response(text="hello")
        assert resp.has_tool_calls is False

    def test_has_tool_calls_true(self):
        tc = ToolCall(id="tc1", name="test", arguments={})
        resp = Response(text="", tool_calls=[tc])
        assert resp.has_tool_calls is True

    def test_defaults(self):
        resp = Response()
        assert resp.text == ""
        assert resp.tool_calls == []
        assert resp.model == ""
        assert resp.usage == {}


class TestToolResult:
    def test_creation(self):
        result = ToolResult(tool_call_id="tc1", output="done")
        assert result.tool_call_id == "tc1"
        assert result.output == "done"
        assert result.is_error is False

    def test_error_result(self):
        result = ToolResult(tool_call_id="tc2", output="failed", is_error=True)
        assert result.is_error is True
