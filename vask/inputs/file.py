"""File input — reads audio from a file path."""

from __future__ import annotations

from pathlib import Path

from vask.config import ProviderConfig
from vask.core.types import AudioChunk


class FileInput:
    """Read audio from a file on disk."""

    def __init__(self, config: ProviderConfig | None = None) -> None:
        self._path: str | None = None
        if config and "path" in config.extra:
            self._path = config.extra["path"]

    def set_path(self, path: str) -> None:
        self._path = path

    async def capture(self) -> AudioChunk:
        if not self._path:
            raise ValueError("No file path set. Call set_path() or set 'path' in config.")
        p = Path(self._path)
        if not p.exists():
            raise FileNotFoundError(f"Audio file not found: {p}")

        data = p.read_bytes()
        suffix = p.suffix.lower().lstrip(".")
        fmt = suffix if suffix in ("wav", "mp3", "ogg", "flac", "m4a", "webm") else "wav"

        return AudioChunk(data=data, format=fmt)

    def stop(self) -> None:
        pass
