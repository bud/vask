"""Pydantic models for the Vask REST API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request to send text to an LLM."""

    text: str = Field(..., description="The text query to send")
    llm: str | None = Field(None, description="LLM provider name")
    system_prompt: str | None = Field(None, description="System prompt override")
    output: str | None = Field(None, description="Output format (terminal, json, clipboard)")


class RecordRequest(BaseModel):
    """Request to record audio and process through pipeline."""

    input: str | None = Field(None, description="Input source name")
    llm: str | None = Field(None, description="LLM provider name")
    system_prompt: str | None = Field(None, description="System prompt override")


class TranscribeRequest(BaseModel):
    """Request to transcribe audio."""

    audio_base64: str = Field(..., description="Base64-encoded audio data")
    format: str = Field("wav", description="Audio format")


class ToolExecRequest(BaseModel):
    """Request to execute a tool directly."""

    tool_name: str = Field(..., description="Name of the tool to execute")
    params: dict = Field(default_factory=dict, description="Tool parameters")


class SessionCreateRequest(BaseModel):
    """Request to create a new conversation session."""

    system_prompt: str | None = Field(None, description="System prompt for this session")
    llm: str | None = Field(None, description="Default LLM for this session")
    plugins: list[str] | None = Field(None, description="Plugins to enable for this session")


class ResponseModel(BaseModel):
    """Standard API response."""

    text: str = ""
    model: str = ""
    usage: dict[str, int] = {}
    tool_calls: list[dict] = []


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: str = ""
    trace_id: str = ""


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = ""
    providers: dict[str, list[str]] = {}
    plugins: list[dict] = []


class SessionInfo(BaseModel):
    """Session information."""

    session_id: str
    created_at: float
    message_count: int
    llm: str | None = None
    system_prompt: str | None = None


class PluginInfo(BaseModel):
    """Plugin information."""

    name: str
    version: str
    description: str
    enabled: bool
    tools: list[str]
    errors: list[str] = []


class APIKeyInfo(BaseModel):
    """API key metadata (no secrets)."""

    key_id: str
    name: str
    role: str
    enabled: bool
    rate_limit: int


class AuditEntry(BaseModel):
    """Audit log entry."""

    timestamp: float
    trace_id: str
    user_id: str
    action: str
    resource: str
    details: dict = {}
    outcome: str
