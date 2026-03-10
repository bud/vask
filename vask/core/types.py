"""Core data types shared across all Vask modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """Raw audio data with metadata."""

    data: bytes
    sample_rate: int = 16000
    channels: int = 1
    format: str = "wav"


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str


@dataclass(slots=True)
class Query:
    """A query to send to an LLM."""

    messages: list[Message]
    system_prompt: str | None = None
    temperature: float = 0.7
    max_tokens: int = 4096


@dataclass(frozen=True, slots=True)
class ToolCall:
    """An LLM-requested tool invocation."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class Response:
    """LLM response, possibly containing tool calls."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Result from executing a tool."""

    tool_call_id: str
    output: str
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class Tool:
    """Tool definition for LLM function calling."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
