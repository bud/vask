"""Core pipeline — Input → Transcribe? → LLM → Tools? → Output."""

from __future__ import annotations

from vask.config import VaskConfig
from vask.core.context import VaskContext
from vask.core.registry import registry
from vask.core.types import AudioChunk, Message, Query, Response, Role, Tool, ToolResult
from vask.inputs.base import InputSource
from vask.llm.base import LLMProvider
from vask.outputs.base import OutputRenderer
from vask.tools.base import ToolProvider
from vask.tools.registry import ToolRegistry
from vask.transcribe.base import TranscriptionProvider


class Pipeline:
    """Orchestrates the full Vask pipeline."""

    def __init__(self, config: VaskConfig) -> None:
        self.config = config
        self.ctx = VaskContext(config=config)
        self.tool_registry = ToolRegistry()

    def _get_input(self, name: str | None = None) -> InputSource:
        name = name or self.config.defaults.get("input", "mic")
        conf = self.config.providers.get(name)
        return registry.create("input", name, conf)

    def _get_transcriber(self, name: str | None = None) -> TranscriptionProvider:
        name = name or self.config.defaults.get("transcription", "openai-whisper")
        conf = self.config.get_provider(name)
        return registry.create("transcription", name, conf)

    def _get_llm(self, name: str | None = None) -> LLMProvider:
        name = name or self.config.defaults.get("llm", "gemini")
        conf = self.config.get_provider(name)
        return registry.create("llm", name, conf)

    def _get_output(self, name: str | None = None) -> OutputRenderer:
        name = name or self.config.defaults.get("output", "terminal")
        conf = self.config.providers.get(name)
        return registry.create("output", name, conf)

    async def record(
        self,
        input_name: str | None = None,
        llm_name: str | None = None,
        output_name: str | None = None,
        system_prompt: str | None = None,
    ) -> Response:
        """Full pipeline: capture audio → transcribe → LLM → output."""
        source = self._get_input(input_name)
        raw = await source.capture()

        if isinstance(raw, str):
            text = raw
        else:
            transcriber = self._get_transcriber()
            text = await transcriber.transcribe(raw)
            self.ctx.log_transcription(input_name or "mic", text)

        return await self.ask(text, llm_name=llm_name, output_name=output_name, system_prompt=system_prompt)

    async def transcribe(
        self,
        input_name: str | None = None,
        output_name: str | None = None,
    ) -> str:
        """Capture audio and transcribe only (no LLM)."""
        source = self._get_input(input_name)
        raw = await source.capture()

        if isinstance(raw, str):
            return raw

        transcriber = self._get_transcriber()
        text = await transcriber.transcribe(raw)
        self.ctx.log_transcription(input_name or "mic", text)

        output = self._get_output(output_name)
        await output.render(Response(text=text))
        return text

    async def ask(
        self,
        text: str,
        llm_name: str | None = None,
        output_name: str | None = None,
        system_prompt: str | None = None,
    ) -> Response:
        """Send text query to LLM with optional tool loop."""
        self.ctx.add_message(Message(role=Role.USER, content=text))

        query = Query(
            messages=list(self.ctx.history),
            system_prompt=system_prompt,
        )

        llm = self._get_llm(llm_name)
        tools = self._get_tool_definitions()

        if tools:
            response = await llm.complete_with_tools(query, tools)
            response = await self._tool_loop(llm, query, tools, response)
        else:
            response = await llm.complete(query)

        # Sync any tool-loop messages back into conversation history
        for msg in query.messages[len(self.ctx.history):]:
            self.ctx.add_message(msg)
        self.ctx.add_message(Message(role=Role.ASSISTANT, content=response.text))

        output = self._get_output(output_name)
        await output.render(response)
        return response

    async def transcribe_audio(self, audio: AudioChunk) -> str:
        """Transcribe an AudioChunk directly."""
        transcriber = self._get_transcriber()
        text = await transcriber.transcribe(audio)
        self.ctx.log_transcription("direct", text)
        return text

    def _get_tool_definitions(self) -> list[Tool]:
        return self.tool_registry.get_tool_definitions()

    async def _tool_loop(
        self,
        llm: LLMProvider,
        query: Query,
        tools: list[Tool],
        response: Response,
        max_iterations: int = 5,
    ) -> Response:
        """Execute tool calls and feed results back to the LLM."""
        iteration = 0
        while response.has_tool_calls and iteration < max_iterations:
            results: list[ToolResult] = []
            for tc in response.tool_calls:
                result = await self.tool_registry.execute(tc.name, tc.arguments, tc.id)
                results.append(result)

            for r in results:
                query.messages.append(Message(role=Role.TOOL, content=f"[{r.tool_call_id}] {r.output}"))

            response = await llm.complete_with_tools(query, tools)
            iteration += 1

        return response
