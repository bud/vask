"""Protocol for audio/text input sources."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from vask.core.types import AudioChunk


@runtime_checkable
class InputSource(Protocol):
    """Any source that produces audio or text."""

    async def capture(self) -> AudioChunk | str:
        """Capture input. Returns AudioChunk for audio or str for text."""
        ...

    def stop(self) -> None:
        """Stop capturing."""
        ...
