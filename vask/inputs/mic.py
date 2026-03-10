"""Microphone input using sounddevice."""

from __future__ import annotations

import asyncio
import io
import wave

import numpy as np
import sounddevice as sd

from vask.config import ProviderConfig
from vask.core.types import AudioChunk

SAMPLE_RATE = 16000
CHANNELS = 1


class MicInput:
    """Capture audio from the default microphone."""

    def __init__(self, config: ProviderConfig | None = None) -> None:
        self._sample_rate = SAMPLE_RATE
        self._channels = CHANNELS
        duration = 5.0
        if config and "duration" in config.extra:
            duration = float(config.extra["duration"])
        self._duration = duration
        self._recording = False

    def _record_sync(self) -> np.ndarray:
        frames = sd.rec(
            int(self._duration * self._sample_rate),
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="int16",
        )
        sd.wait()
        return np.asarray(frames)

    async def capture(self) -> AudioChunk:
        self._recording = True
        loop = asyncio.get_event_loop()
        frames = await loop.run_in_executor(None, self._record_sync)
        self._recording = False

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self._channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self._sample_rate)
            wf.writeframes(frames.tobytes())

        return AudioChunk(
            data=buf.getvalue(),
            sample_rate=self._sample_rate,
            channels=self._channels,
            format="wav",
        )

    def stop(self) -> None:
        if self._recording:
            sd.stop()
            self._recording = False
