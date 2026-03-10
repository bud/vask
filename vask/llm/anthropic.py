"""Anthropic Claude LLM provider."""

from __future__ import annotations

import json

import anthropic

from vask.config import ProviderConfig
from vask.core.types import Query, Response, Tool, ToolCall


class AnthropicLLM:
    """LLM provider using the Anthropic API."""

    def __init__(self, config: ProviderConfig) -> None:
        api_key = config.api_key
        if not api_key:
            raise ValueError(f"API key not found. Set the {config.api_key_env} environment variable.")
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = config.model or "claude-sonnet-4-20250514"

    def _to_messages(self, query: Query) -> list[dict]:
        msgs: list[dict] = []
        for m in query.messages:
            role = m.role.value
            if role == "system":
                continue  # system goes via kwargs, not messages
            if role == "tool":
                role = "user"
            # Anthropic requires alternating user/assistant. Merge consecutive same-role messages.
            if msgs and msgs[-1]["role"] == role:
                msgs[-1]["content"] += "\n" + m.content
            else:
                msgs.append({"role": role, "content": m.content})
        return msgs

    def _to_anthropic_tools(self, tools: list[Tool]) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in tools
        ]

    def _parse_response(self, resp: anthropic.types.Message) -> Response:
        text_parts = []
        tool_calls = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else json.loads(block.input),
                    )
                )
        usage = {
            "prompt_tokens": resp.usage.input_tokens,
            "completion_tokens": resp.usage.output_tokens,
        }
        return Response(
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            model=resp.model,
            usage=usage,
        )

    async def complete(self, query: Query) -> Response:
        kwargs: dict = {
            "model": self._model,
            "messages": self._to_messages(query),
            "max_tokens": query.max_tokens,
        }
        if query.system_prompt:
            kwargs["system"] = query.system_prompt
        resp = await self._client.messages.create(**kwargs)
        return self._parse_response(resp)

    async def complete_with_tools(self, query: Query, tools: list[Tool]) -> Response:
        kwargs: dict = {
            "model": self._model,
            "messages": self._to_messages(query),
            "max_tokens": query.max_tokens,
            "tools": self._to_anthropic_tools(tools),
        }
        if query.system_prompt:
            kwargs["system"] = query.system_prompt
        resp = await self._client.messages.create(**kwargs)
        return self._parse_response(resp)
