"""OpenAI / OpenRouter LLM provider."""

from __future__ import annotations

import json

import openai

from vask.config import ProviderConfig
from vask.core.types import Query, Response, Role, Tool, ToolCall


class OpenAILLM:
    """LLM provider using OpenAI-compatible APIs (also works for OpenRouter)."""

    def __init__(self, config: ProviderConfig) -> None:
        api_key = config.api_key
        if not api_key:
            raise ValueError(f"API key not found. Set the {config.api_key_env} environment variable.")
        kwargs: dict = {"api_key": api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self._client = openai.AsyncOpenAI(**kwargs)
        self._model = config.model or "gpt-4o"

    def _to_messages(self, query: Query) -> list[dict]:
        msgs = []
        if query.system_prompt:
            msgs.append({"role": "system", "content": query.system_prompt})
        for m in query.messages:
            role = m.role.value
            if role == Role.TOOL.value:
                role = "user"  # flatten tool results as user messages
            msgs.append({"role": role, "content": m.content})
        return msgs

    def _to_openai_tools(self, tools: list[Tool]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    def _parse_response(self, resp: openai.types.chat.ChatCompletion) -> Response:
        choice = resp.choices[0]
        text = choice.message.content or ""
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )
        usage = {}
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            }
        return Response(text=text, tool_calls=tool_calls, model=resp.model, usage=usage)

    async def complete(self, query: Query) -> Response:
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=self._to_messages(query),
            temperature=query.temperature,
            max_tokens=query.max_tokens,
        )
        return self._parse_response(resp)

    async def complete_with_tools(self, query: Query, tools: list[Tool]) -> Response:
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=self._to_messages(query),
            temperature=query.temperature,
            max_tokens=query.max_tokens,
            tools=self._to_openai_tools(tools),
        )
        return self._parse_response(resp)
