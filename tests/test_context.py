"""Tests for vask.core.context.VaskContext."""

from vask.core.context import VaskContext
from vask.core.types import Message, Role


class TestVaskContext:
    def test_add_message_appends_to_history(self, mock_config):
        ctx = VaskContext(config=mock_config)
        msg = Message(role=Role.USER, content="hello")
        ctx.add_message(msg)
        assert len(ctx.history) == 1
        assert ctx.history[0] is msg

    def test_add_multiple_messages(self, mock_config):
        ctx = VaskContext(config=mock_config)
        ctx.add_message(Message(role=Role.USER, content="q1"))
        ctx.add_message(Message(role=Role.ASSISTANT, content="a1"))
        assert len(ctx.history) == 2
        assert ctx.history[0].role == Role.USER
        assert ctx.history[1].role == Role.ASSISTANT

    def test_log_transcription_appends_to_log(self, mock_config):
        ctx = VaskContext(config=mock_config)
        ctx.log_transcription("mic", "hello world")
        assert len(ctx.transcription_log) == 1
        assert ctx.transcription_log[0] == {"source": "mic", "text": "hello world"}

    def test_log_transcription_multiple(self, mock_config):
        ctx = VaskContext(config=mock_config)
        ctx.log_transcription("mic", "first")
        ctx.log_transcription("file", "second")
        assert len(ctx.transcription_log) == 2
        assert ctx.transcription_log[1]["source"] == "file"

    def test_clear_history_empties_history(self, mock_config):
        ctx = VaskContext(config=mock_config)
        ctx.add_message(Message(role=Role.USER, content="hello"))
        ctx.add_message(Message(role=Role.ASSISTANT, content="hi"))
        assert len(ctx.history) == 2
        ctx.clear_history()
        assert len(ctx.history) == 0

    def test_clear_history_does_not_affect_transcription_log(self, mock_config):
        ctx = VaskContext(config=mock_config)
        ctx.add_message(Message(role=Role.USER, content="hello"))
        ctx.log_transcription("mic", "hello")
        ctx.clear_history()
        assert len(ctx.history) == 0
        assert len(ctx.transcription_log) == 1
