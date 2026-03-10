"""Google Gemini LLM provider."""

from __future__ import annotations

import uuid

from google import genai
from google.genai import types as genai_types

from vask.config import ProviderConfig
from vask.core.types import Query, Response, Tool, ToolCall


class GeminiLLM:
    """LLM provider using the Google GenAI SDK."""

    def __init__(self, config: ProviderConfig) -> None:
        api_key = config.api_key
        if not api_key:
            raise ValueError(f"API key not found. Set the {config.api_key_env} environment variable.")
        self._client = genai.Client(api_key=api_key)
        self._model = config.model or "gemini-2.0-flash"

    def _to_contents(self, query: Query) -> list[genai_types.Content]:
        contents = []
        for m in query.messages:
            role = "user" if m.role.value in ("user", "system", "tool") else "model"
            contents.append(genai_types.Content(role=role, parts=[genai_types.Part(text=m.content)]))
        return contents

    def _to_gemini_tools(self, tools: list[Tool]) -> list[genai_types.Tool]:
        declarations = []
        for t in tools:
            declarations.append(
                genai_types.FunctionDeclaration(
                    name=t.name,
                    description=t.description,
                    parameters=t.parameters,
                )
            )
        return [genai_types.Tool(function_declarations=declarations)]

    def _parse_response(self, resp) -> Response:
        text_parts = []
        tool_calls = []

        for candidate in resp.candidates:
            for part in candidate.content.parts:
                if part.text:
                    text_parts.append(part.text)
                elif part.function_call:
                    fc = part.function_call
                    tool_calls.append(
                        ToolCall(
                            id=f"{fc.name}-{uuid.uuid4().hex[:8]}",
                            name=fc.name,
                            arguments=dict(fc.args) if fc.args else {},
                        )
                    )

        usage = {}
        if resp.usage_metadata:
            usage = {
                "prompt_tokens": resp.usage_metadata.prompt_token_count or 0,
                "completion_tokens": resp.usage_metadata.candidates_token_count or 0,
            }

        return Response(
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            model=self._model,
            usage=usage,
        )

    async def complete(self, query: Query) -> Response:
        config = genai_types.GenerateContentConfig(
            temperature=query.temperature,
            max_output_tokens=query.max_tokens,
        )
        if query.system_prompt:
            config.system_instruction = query.system_prompt

        resp = await self._client.aio.models.generate_content(
            model=self._model,
            contents=self._to_contents(query),
            config=config,
        )
        return self._parse_response(resp)

    async def complete_with_tools(self, query: Query, tools: list[Tool]) -> Response:
        config = genai_types.GenerateContentConfig(
            temperature=query.temperature,
            max_output_tokens=query.max_tokens,
            tools=self._to_gemini_tools(tools),
        )
        if query.system_prompt:
            config.system_instruction = query.system_prompt

        resp = await self._client.aio.models.generate_content(
            model=self._model,
            contents=self._to_contents(query),
            config=config,
        )
        return self._parse_response(resp)
