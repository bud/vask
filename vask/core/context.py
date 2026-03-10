"""Runtime context — holds config, providers, and conversation history."""

from __future__ import annotations

from dataclasses import dataclass, field

from vask.config import VaskConfig
from vask.core.types import Message


@dataclass(slots=True)
class VaskContext:
    """Runtime state for a Vask session."""

    config: VaskConfig
    history: list[Message] = field(default_factory=list)
    transcription_log: list[dict[str, str]] = field(default_factory=list)

    def add_message(self, message: Message) -> None:
        self.history.append(message)

    def log_transcription(self, audio_source: str, text: str) -> None:
        self.transcription_log.append({"source": audio_source, "text": text})

    def clear_history(self) -> None:
        self.history.clear()
