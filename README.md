# Vask

Voice-to-Action framework — pluggable audio capture, transcription, LLM reasoning, and tool execution. Runs standalone CLI or as an MCP server plugin.

## Quick Start

```bash
pip install -e .
vask --help
```

## MCP Server

Add to your MCP client config:

```json
{
  "mcpServers": {
    "vask": {
      "command": "vask",
      "args": ["mcp"]
    }
  }
}
```

## Configuration

Config lives at `~/.config/vask/config.toml`. Set your API keys as environment variables:

```bash
export GEMINI_API_KEY="..."
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```
