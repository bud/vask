"""Clipboard input — reads text from the system clipboard, skipping transcription."""

from __future__ import annotations

import subprocess

from vask.config import ProviderConfig


class ClipboardInput:
    """Read text from the system clipboard."""

    def __init__(self, config: ProviderConfig | None = None) -> None:
        pass

    async def capture(self) -> str:
        result = subprocess.run(["pbpaste"], capture_output=True, text=True, check=True)
        text = result.stdout.strip()
        if not text:
            raise ValueError("Clipboard is empty")
        return text

    def stop(self) -> None:
        pass
