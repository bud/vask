"""Protocol for LLM providers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from vask.core.types import Query, Response, Tool


@runtime_checkable
class LLMProvider(Protocol):
    """Any LLM backend (cloud or local)."""

    async def complete(self, query: Query) -> Response: ...

    async def complete_with_tools(self, query: Query, tools: list[Tool]) -> Response: ...
