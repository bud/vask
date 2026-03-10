"""Protocol for output renderers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from vask.core.types import Response


@runtime_checkable
class OutputRenderer(Protocol):
    """Renders an LLM response to a destination."""

    async def render(self, response: Response) -> None: ...
