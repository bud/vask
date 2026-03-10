"""Protocol for transcription providers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from vask.core.types import AudioChunk


@runtime_checkable
class TranscriptionProvider(Protocol):
    """Converts audio to text."""

    async def transcribe(self, audio: AudioChunk) -> str: ...
