"""MCP server — exposes Vask as tools and resources via Model Context Protocol."""

from __future__ import annotations

import asyncio
import json

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, TextContent, Tool

from vask.config import VaskConfig
from vask.core.pipeline import Pipeline
from vask.core.registry import register_defaults
from vask.core.types import AudioChunk

server = Server("vask")
_pipeline: Pipeline | None = None
_config: VaskConfig | None = None


def _get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialized")
    return _pipeline


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="vask_ask",
            description="Send a text query to any configured LLM provider. Supports Gemini, Claude, OpenAI, OpenRouter, and local models.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The text query to send to the LLM"},
                    "llm": {
                        "type": "string",
                        "description": "LLM provider name (gemini, claude, openai, openrouter). Uses config default if omitted.",
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "Optional system prompt to set context for the LLM",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="vask_record",
            description="Capture audio from the microphone, transcribe it, then send to an LLM for processing.",
            inputSchema={
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Input source (mic, clipboard). Default: mic",
                    },
                    "llm": {"type": "string", "description": "LLM provider to use"},
                    "system_prompt": {"type": "string", "description": "System prompt for the LLM"},
                },
            },
        ),
        Tool(
            name="vask_transcribe",
            description="Capture audio and transcribe it to text (no LLM processing).",
            inputSchema={
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Input source (mic, file). Default: mic",
                    },
                },
            },
        ),
        Tool(
            name="vask_transcribe_audio",
            description="Transcribe raw audio data (base64-encoded) to text.",
            inputSchema={
                "type": "object",
                "properties": {
                    "audio_base64": {
                        "type": "string",
                        "description": "Base64-encoded audio data",
                    },
                    "format": {
                        "type": "string",
                        "description": "Audio format (wav, mp3, ogg, etc.)",
                        "default": "wav",
                    },
                },
                "required": ["audio_base64"],
            },
        ),
        Tool(
            name="vask_providers",
            description="List all available providers and their configuration status.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    pipeline = _get_pipeline()

    if name == "vask_ask":
        response = await pipeline.ask(
            arguments["query"],
            llm_name=arguments.get("llm"),
            output_name="json",
            system_prompt=arguments.get("system_prompt"),
        )
        return [TextContent(type="text", text=response.text)]

    elif name == "vask_record":
        response = await pipeline.record(
            input_name=arguments.get("input"),
            llm_name=arguments.get("llm"),
            output_name="json",
            system_prompt=arguments.get("system_prompt"),
        )
        return [TextContent(type="text", text=response.text)]

    elif name == "vask_transcribe":
        text = await pipeline.transcribe(
            input_name=arguments.get("input"),
            output_name="json",
        )
        return [TextContent(type="text", text=text)]

    elif name == "vask_transcribe_audio":
        import base64

        MAX_AUDIO_SIZE = 25 * 1024 * 1024  # 25 MB
        raw_b64 = arguments["audio_base64"]
        if len(raw_b64) > MAX_AUDIO_SIZE * 4 // 3:
            return [TextContent(type="text", text="Error: audio data exceeds 25 MB limit.")]
        audio_data = base64.b64decode(raw_b64)
        if len(audio_data) > MAX_AUDIO_SIZE:
            return [TextContent(type="text", text="Error: audio data exceeds 25 MB limit.")]
        fmt = arguments.get("format", "wav")
        audio = AudioChunk(data=audio_data, format=fmt)
        text = await pipeline.transcribe_audio(audio)
        return [TextContent(type="text", text=text)]

    elif name == "vask_providers":
        cfg = _get_pipeline().config
        info = {
            "defaults": cfg.defaults,
            "providers": {
                pname: {
                    "type": p.type,
                    "model": p.model,
                    "has_api_key": bool(p.api_key),
                }
                for pname, p in cfg.providers.items()
            },
        }
        return [TextContent(type="text", text=json.dumps(info, indent=2))]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


@server.list_resources()
async def list_resources() -> list[Resource]:
    return [
        Resource(
            uri="vask://transcription-history",
            name="Transcription History",
            description="Log of all transcriptions performed in this session",
            mimeType="application/json",
        ),
        Resource(
            uri="vask://config",
            name="Vask Configuration",
            description="Current Vask configuration and provider status",
            mimeType="application/json",
        ),
    ]


@server.read_resource()
async def read_resource(uri: str) -> str:
    pipeline = _get_pipeline()

    if str(uri) == "vask://transcription-history":
        return json.dumps(pipeline.ctx.transcription_log, indent=2)

    elif str(uri) == "vask://config":
        cfg = pipeline.config
        return json.dumps(
            {
                "defaults": cfg.defaults,
                "providers": {
                    pname: {"type": p.type, "model": p.model, "has_api_key": bool(p.api_key)}
                    for pname, p in cfg.providers.items()
                },
            },
            indent=2,
        )

    raise ValueError(f"Unknown resource: {uri}")


def run_server(config: VaskConfig, transport: str = "stdio") -> None:
    """Start the MCP server."""
    global _pipeline, _config
    register_defaults()
    _config = config
    _pipeline = Pipeline(config)

    if transport == "stdio":
        asyncio.run(_run_stdio())
    else:
        raise ValueError(f"Unsupported transport: {transport}. Currently only 'stdio' is supported.")


async def _run_stdio() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
