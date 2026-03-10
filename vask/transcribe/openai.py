"""OpenAI Whisper transcription provider."""

from __future__ import annotations

import io

import openai

from vask.config import ProviderConfig
from vask.core.types import AudioChunk


class OpenAIWhisper:
    """Transcribe audio using the OpenAI Whisper API."""

    def __init__(self, config: ProviderConfig) -> None:
        api_key = config.api_key
        if not api_key:
            raise ValueError(
                f"API key not found. Set the {config.api_key_env} environment variable."
            )
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._model = config.model or "whisper-1"

    async def transcribe(self, audio: AudioChunk) -> str:
        ext = audio.format if audio.format != "wav" else "wav"
        filename = f"audio.{ext}"

        audio_file = io.BytesIO(audio.data)
        audio_file.name = filename

        response = await self._client.audio.transcriptions.create(
            model=self._model,
            file=audio_file,
        )
        return response.text
