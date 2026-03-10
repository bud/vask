# Vask

**Voice-to-Action AI Framework** — enterprise-ready automation that connects voice input, LLM reasoning, and tool execution into streamlined business workflows.

Built by [Techforcement](https://techforcement.com) to help growing businesses automate operations and scale efficiently.

## What Vask Does

Vask turns natural language — spoken or typed — into automated business actions. Instead of switching between tools, writing scripts, or waiting on manual processes, teams describe what they need and Vask handles the rest.

**For business operations:**
- Automate CRM updates, customer follow-ups, and sales logging via voice
- Trigger multi-step workflows across Slack, Google Workspace, and databases
- Monitor services and respond to incidents with AI-guided remediation
- Process inbound webhooks from Stripe, GitHub, Twilio into automated actions

**For technical teams:**
- Multi-provider LLM support (Gemini, Claude, OpenAI, OpenRouter)
- Plugin system for Slack, Google Workspace, SQL databases
- YAML-driven workflow engine with conditional branching and human approval gates
- REST API, CLI, and MCP server deployment options
- RBAC authentication, rate limiting, and audit logging

## Quick Start

```bash
pip install -e .
vask --help
```

### Core Commands

```bash
vask ask "Summarize today's sales pipeline"       # Text query to LLM
vask record                                        # Voice → transcribe → LLM
vask transcribe                                    # Voice → text only
vask serve                                         # Start REST API server
vask mcp                                           # Start MCP server
vask plugins                                       # List available plugins
```

## Configuration

Config lives at `~/.config/vask/config.toml`:

```toml
[defaults]
input = "mic"
transcription = "openai-whisper"
llm = "gemini"
output = "terminal"

[providers.gemini]
type = "llm"
api_key_env = "GEMINI_API_KEY"
model = "gemini-2.0-flash"

[providers.claude]
type = "llm"
api_key_env = "ANTHROPIC_API_KEY"
model = "claude-sonnet-4-20250514"

[server]
host = "0.0.0.0"
port = 8420
cors_origins = ["*"]
```

Set API keys as environment variables:

```bash
export GEMINI_API_KEY="..."
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

## Workflow Examples

Vask workflows are YAML-defined automation pipelines. See the [examples/](examples/) directory:

| Workflow | Use Case |
|----------|----------|
| `crm_call_log.yaml` | Log sales calls → extract CRM data → notify Slack |
| `incident_response.yaml` | Check service health → AI analysis → approve → notify |
| `client_onboarding.yaml` | Welcome email → provision workspace → schedule kickoff |
| `lead_qualification.yaml` | Score inbound leads → route to sales → update CRM |
| `weekly_report.yaml` | Pull metrics → generate summary → distribute to stakeholders |

## API Server

```bash
vask serve --port 8420
```

**Key endpoints:**
- `POST /v1/ask` — Query LLM with text
- `POST /v1/transcribe` — Transcribe audio (base64)
- `POST /v1/sessions` — Manage conversation sessions
- `POST /v1/workflows/{id}/run` — Execute automation workflows
- `POST /v1/tools/execute` — Run tools directly
- `POST /v1/webhooks/{id}/receive` — Inbound webhook receiver
- `GET /health` — Health check with provider status

## MCP Server

Integrate Vask as tools in Claude or other MCP-compatible clients:

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

## Plugins

| Plugin | Tools |
|--------|-------|
| **Slack** | Send messages, search, react, set status, list channels |
| **Google Workspace** | Sheets, Docs, Drive, Gmail operations |
| **SQL** | Execute queries against SQLite, PostgreSQL, MySQL |

Enable plugins via config or at runtime through the API.

## Deployment

```bash
# Docker
docker compose up -d

# Direct
vask serve --host 0.0.0.0 --port 8420
```

## Architecture

```
Voice/Text Input → Transcription (Whisper) → LLM Reasoning → Tool Execution → Output
                                                    ↕
                                          Workflow Engine
                                          Plugin System
                                          Webhook Handler
```

Modular provider pattern — swap any component (input, transcription, LLM, output, tools) without changing the pipeline.

## License

MIT
