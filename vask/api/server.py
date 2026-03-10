"""FastAPI server — REST + WebSocket API for Vask."""

from __future__ import annotations

import base64
import time
import uuid
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware

from vask import __version__
from vask.api.models import (
    APIKeyInfo,
    AskRequest,
    AuditEntry,
    HealthResponse,
    PluginInfo,
    ResponseModel,
    SessionCreateRequest,
    SessionInfo,
    ToolExecRequest,
    TranscribeRequest,
)
from vask.auth import AuthManager, Permission
from vask.config import VaskConfig, ensure_config
from vask.core.pipeline import Pipeline
from vask.core.registry import register_defaults
from vask.core.types import AudioChunk, Message, Role
from vask.logging import (
    Span,
    audit,
    get_logger,
    new_trace_id,
    set_trace_context,
    setup_logging,
)
from vask.plugins.loader import PluginLoader
from vask.tools.registry import ToolRegistry
from vask.webhooks import WebhookEvent, WebhookRegistry, WebhookSource, extract_text_from_webhook
from vask.workflows import WorkflowEngine

logger = get_logger("api")


# ── Session store ──────────────────────────────────────────────────────

class SessionStore:
    """In-memory session management (swap for Redis/Postgres in production)."""

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}

    def create(
        self,
        system_prompt: str | None = None,
        llm: str | None = None,
    ) -> str:
        session_id = uuid.uuid4().hex[:16]
        self._sessions[session_id] = {
            "id": session_id,
            "created_at": time.time(),
            "history": [],
            "system_prompt": system_prompt,
            "llm": llm,
        }
        return session_id

    def get(self, session_id: str) -> dict[str, Any] | None:
        return self._sessions.get(session_id)

    def add_message(self, session_id: str, role: str, content: str) -> None:
        session = self._sessions.get(session_id)
        if session:
            session["history"].append({"role": role, "content": content})

    def delete(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def list_sessions(self) -> list[dict[str, Any]]:
        return [
            {
                "session_id": s["id"],
                "created_at": s["created_at"],
                "message_count": len(s["history"]),
                "llm": s.get("llm"),
                "system_prompt": s.get("system_prompt"),
            }
            for s in self._sessions.values()
        ]


# ── App factory ────────────────────────────────────────────────────────

def create_app(config: VaskConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    setup_logging(level="INFO", json_output=True)
    register_defaults()

    if config is None:
        config = ensure_config()

    app = FastAPI(
        title="Vask API",
        description=(
            "Voice-to-Action framework — REST API for audio capture, "
            "transcription, LLM reasoning, and tool execution."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Shared state
    auth_manager = AuthManager()
    session_store = SessionStore()
    tool_registry = ToolRegistry()

    # Load plugins
    plugin_loader = PluginLoader(tool_registry)
    plugin_loader.load_all()

    # Register built-in tools
    from vask.tools.http import HttpTool
    from vask.tools.shell import ShellTool

    http_conf = config.providers.get("http")
    shell_conf = config.providers.get("shell")
    tool_registry.register(HttpTool(http_conf))
    tool_registry.register(ShellTool(shell_conf))

    def _make_pipeline() -> Pipeline:
        pipeline = Pipeline(config)
        pipeline.tool_registry = tool_registry
        return pipeline

    # ── Auth dependency ────────────────────────────────────────────────

    async def get_current_key(authorization: str | None = Header(None)):
        """Extract and validate API key from Authorization header."""
        # If no keys are configured, allow all requests (dev mode)
        if not auth_manager._keys:
            return None

        if not authorization:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing Authorization header. Use 'Bearer <api_key>'",
            )

        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Authorization header format. Use 'Bearer <api_key>'",
            )

        result = auth_manager.authenticate(token)
        if not result.authenticated:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=result.error,
            )

        if not auth_manager.check_rate_limit(result.key):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
            )

        return result.key

    def require_permission(permission: Permission):
        """Dependency that checks a specific permission."""

        async def checker(key=Depends(get_current_key)):
            if key is None:
                return  # dev mode, no auth configured
            if not auth_manager.authorize(key, permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Missing permission: {permission.value}",
                )
            return key

        return checker

    # ── Middleware ──────────────────────────────────────────────────────

    @app.middleware("http")
    async def trace_middleware(request, call_next):
        trace_id = request.headers.get("X-Trace-ID", new_trace_id())
        set_trace_context(trace_id)
        with Span("http_request") as span:
            span.attributes["method"] = request.method
            span.attributes["path"] = request.url.path
            response = await call_next(request)
            span.attributes["status_code"] = response.status_code
            response.headers["X-Trace-ID"] = trace_id
            return response

    # ── Health ─────────────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health():
        from vask.core.registry import registry

        providers = {}
        for cat in ["input", "transcription", "llm", "output", "tool"]:
            providers[cat] = registry.list_providers(cat)

        return HealthResponse(
            status="ok",
            version=__version__,
            providers=providers,
            plugins=plugin_loader.list_plugins(),
        )

    # ── Pipeline endpoints ─────────────────────────────────────────────

    @app.post("/v1/ask", response_model=ResponseModel, tags=["pipeline"])
    async def ask(
        req: AskRequest,
        _key=Depends(require_permission(Permission.PIPELINE_ASK)),
    ):
        with Span("api_ask") as span:
            span.attributes["llm"] = req.llm
            pipeline = _make_pipeline()
            response = await pipeline.ask(
                req.text,
                llm_name=req.llm,
                output_name="json",
                system_prompt=req.system_prompt,
            )
            audit.log("pipeline", "ask", {"text_length": len(req.text)})
            return ResponseModel(
                text=response.text,
                model=response.model,
                usage=response.usage,
                tool_calls=[
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in response.tool_calls
                ],
            )

    @app.post("/v1/transcribe", response_model=ResponseModel, tags=["pipeline"])
    async def transcribe(
        req: TranscribeRequest,
        _key=Depends(require_permission(Permission.PIPELINE_TRANSCRIBE)),
    ):
        with Span("api_transcribe") as span:
            audio_bytes = base64.b64decode(req.audio_base64)
            span.attributes["audio_size"] = len(audio_bytes)

            if len(audio_bytes) > 25 * 1024 * 1024:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Audio exceeds 25 MB limit",
                )

            pipeline = _make_pipeline()
            chunk = AudioChunk(data=audio_bytes, format=req.format)
            text = await pipeline.transcribe_audio(chunk)
            audit.log("pipeline", "transcribe", {"audio_size": len(audio_bytes)})
            return ResponseModel(text=text)

    # ── Tool endpoints ─────────────────────────────────────────────────

    @app.get("/v1/tools", tags=["tools"])
    async def list_tools(_key=Depends(get_current_key)):
        definitions = tool_registry.get_tool_definitions()
        return [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in definitions
        ]

    @app.post("/v1/tools/execute", tags=["tools"])
    async def execute_tool(
        req: ToolExecRequest,
        _key=Depends(require_permission(Permission.TOOLS_EXECUTE)),
    ):
        with Span("tool_execute") as span:
            span.attributes["tool"] = req.tool_name
            result = await tool_registry.execute(
                req.tool_name, req.params, tool_call_id=f"api_{uuid.uuid4().hex[:8]}"
            )
            audit.log("tool", req.tool_name, {"params": list(req.params.keys())})
            return {
                "tool_call_id": result.tool_call_id,
                "output": result.output,
                "is_error": result.is_error,
            }

    # ── Session endpoints ──────────────────────────────────────────────

    @app.post("/v1/sessions", response_model=SessionInfo, tags=["sessions"])
    async def create_session(
        req: SessionCreateRequest,
        _key=Depends(require_permission(Permission.PIPELINE_ASK)),
    ):
        session_id = session_store.create(
            system_prompt=req.system_prompt,
            llm=req.llm,
        )
        session = session_store.get(session_id)
        return SessionInfo(
            session_id=session_id,
            created_at=session["created_at"],
            message_count=0,
            llm=req.llm,
            system_prompt=req.system_prompt,
        )

    @app.get("/v1/sessions", tags=["sessions"])
    async def list_sessions(_key=Depends(get_current_key)):
        return session_store.list_sessions()

    @app.post("/v1/sessions/{session_id}/ask", response_model=ResponseModel, tags=["sessions"])
    async def session_ask(
        session_id: str,
        req: AskRequest,
        _key=Depends(require_permission(Permission.PIPELINE_ASK)),
    ):
        session = session_store.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        with Span("session_ask") as span:
            span.attributes["session_id"] = session_id
            pipeline = _make_pipeline()

            # Restore conversation history
            for msg in session["history"]:
                pipeline.ctx.add_message(
                    Message(role=Role(msg["role"]), content=msg["content"])
                )

            llm = req.llm or session.get("llm")
            system_prompt = req.system_prompt or session.get("system_prompt")

            response = await pipeline.ask(
                req.text,
                llm_name=llm,
                output_name="json",
                system_prompt=system_prompt,
            )

            # Persist messages
            session_store.add_message(session_id, "user", req.text)
            session_store.add_message(session_id, "assistant", response.text)

            return ResponseModel(
                text=response.text,
                model=response.model,
                usage=response.usage,
                tool_calls=[
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in response.tool_calls
                ],
            )

    @app.delete("/v1/sessions/{session_id}", tags=["sessions"])
    async def delete_session(
        session_id: str,
        _key=Depends(get_current_key),
    ):
        if not session_store.delete(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        return {"deleted": True}

    # ── Plugin endpoints ───────────────────────────────────────────────

    @app.get("/v1/plugins", response_model=list[PluginInfo], tags=["plugins"])
    async def list_plugins(_key=Depends(get_current_key)):
        return [PluginInfo(**p) for p in plugin_loader.list_plugins()]

    @app.post("/v1/plugins/{name}/enable", tags=["plugins"])
    async def enable_plugin(
        name: str,
        _key=Depends(require_permission(Permission.ADMIN_CONFIG)),
    ):
        if not plugin_loader.enable_plugin(name):
            raise HTTPException(status_code=404, detail="Plugin not found")
        return {"enabled": True}

    @app.post("/v1/plugins/{name}/disable", tags=["plugins"])
    async def disable_plugin(
        name: str,
        _key=Depends(require_permission(Permission.ADMIN_CONFIG)),
    ):
        if not plugin_loader.disable_plugin(name):
            raise HTTPException(status_code=404, detail="Plugin not found")
        return {"disabled": True}

    # ── Admin endpoints ────────────────────────────────────────────────

    @app.get("/v1/admin/keys", response_model=list[APIKeyInfo], tags=["admin"])
    async def list_api_keys(
        _key=Depends(require_permission(Permission.ADMIN_USERS)),
    ):
        return [APIKeyInfo(**k) for k in auth_manager.list_keys()]

    @app.get("/v1/admin/audit", response_model=list[AuditEntry], tags=["admin"])
    async def get_audit_log(
        limit: int = 100,
        _key=Depends(require_permission(Permission.ADMIN_AUDIT)),
    ):
        return [AuditEntry(**e) for e in audit.get_entries(limit)]

    @app.get("/v1/config", tags=["admin"])
    async def get_config(
        _key=Depends(require_permission(Permission.ADMIN_CONFIG)),
    ):
        return {
            "defaults": config.defaults,
            "providers": {
                name: {
                    "type": p.type,
                    "model": p.model,
                    "has_key": bool(p.api_key),
                }
                for name, p in config.providers.items()
            },
        }

    # ── Webhook endpoints ────────────────────────────────────────────────

    webhook_registry = WebhookRegistry()

    # Wire up the "ask" handler for webhooks
    async def _webhook_ask_handler(event: WebhookEvent) -> str:
        webhook = webhook_registry.get(event.webhook_id)
        if not webhook:
            return "Unknown webhook"
        text = extract_text_from_webhook(
            webhook.source,
            event.body if isinstance(event.body, dict) else {},
        )
        if not text:
            return "No actionable text found in webhook payload"
        pipeline = _make_pipeline()
        cfg = webhook.action_config
        response = await pipeline.ask(
            text,
            llm_name=cfg.get("llm"),
            output_name="json",
            system_prompt=cfg.get("system_prompt"),
        )
        return response.text

    webhook_registry.set_handler("ask", _webhook_ask_handler)

    @app.post("/v1/webhooks", tags=["webhooks"])
    async def create_webhook(
        name: str,
        source: str = "generic",
        secret: str = "",
        action: str = "ask",
        _key=Depends(require_permission(Permission.ADMIN_CONFIG)),
    ):
        try:
            src = WebhookSource(source)
        except ValueError:
            src = WebhookSource.GENERIC
        config = webhook_registry.register(name=name, source=src, secret=secret, action=action)
        return {"id": config.id, "name": config.name, "url": f"/v1/webhooks/{config.id}/receive"}

    @app.get("/v1/webhooks", tags=["webhooks"])
    async def list_webhooks(_key=Depends(get_current_key)):
        return webhook_registry.list_webhooks()

    @app.post("/v1/webhooks/{webhook_id}/receive", tags=["webhooks"])
    async def receive_webhook(webhook_id: str, request: Any):
        """Inbound webhook receiver — no auth required (verified by signature)."""
        from starlette.requests import Request

        req: Request = request
        raw_body = await req.body()
        headers = dict(req.headers)

        webhook = webhook_registry.get(webhook_id)
        if not webhook:
            raise HTTPException(status_code=404, detail="Unknown webhook")
        if not webhook.enabled:
            raise HTTPException(status_code=403, detail="Webhook is disabled")

        # Verify signature
        if not webhook_registry.verify_signature(webhook, raw_body, headers):
            raise HTTPException(status_code=401, detail="Invalid signature")

        # Parse body
        import json as json_lib
        try:
            body = json_lib.loads(raw_body)
        except (json_lib.JSONDecodeError, UnicodeDecodeError):
            body = raw_body.decode("utf-8", errors="replace")

        event = WebhookEvent(
            webhook_id=webhook_id,
            source=webhook.source,
            headers=headers,
            body=body,
            raw_body=raw_body,
        )

        result = await webhook_registry.dispatch(event)
        return {"event_id": event.id, "result": result[:1000]}

    @app.get("/v1/webhooks/events", tags=["webhooks"])
    async def get_webhook_events(
        limit: int = 50,
        _key=Depends(get_current_key),
    ):
        return webhook_registry.get_event_log(limit)

    # ── Workflow endpoints ─────────────────────────────────────────────

    workflow_engine = WorkflowEngine(
        pipeline=_make_pipeline(),
        tool_registry=tool_registry,
    )

    # Wire up workflow handler for webhooks
    async def _webhook_workflow_handler(event: WebhookEvent) -> str:
        webhook = webhook_registry.get(event.webhook_id)
        if not webhook:
            return "Unknown webhook"
        workflow_id = webhook.action_config.get("workflow_id", "")
        if not workflow_id:
            return "No workflow_id in webhook config"
        text = extract_text_from_webhook(
            webhook.source,
            event.body if isinstance(event.body, dict) else {},
        )
        run = await workflow_engine.execute(workflow_id, variables={"input": text})
        return f"Workflow run {run.id}: {run.status.value}"

    webhook_registry.set_handler("workflow", _webhook_workflow_handler)

    @app.post("/v1/workflows", tags=["workflows"])
    async def create_workflow(
        data: dict[str, Any],
        _key=Depends(require_permission(Permission.ADMIN_CONFIG)),
    ):
        workflow = workflow_engine.register_from_dict(data)
        return {
            "id": workflow.id,
            "name": workflow.name,
            "steps_count": len(workflow.steps),
        }

    @app.get("/v1/workflows", tags=["workflows"])
    async def list_workflows(_key=Depends(get_current_key)):
        return workflow_engine.list_workflows()

    @app.get("/v1/workflows/{workflow_id}", tags=["workflows"])
    async def get_workflow(workflow_id: str, _key=Depends(get_current_key)):
        wf = workflow_engine.get_workflow(workflow_id)
        if not wf:
            raise HTTPException(status_code=404, detail="Workflow not found")
        return {
            "id": wf.id,
            "name": wf.name,
            "description": wf.description,
            "trigger": wf.trigger,
            "steps": [
                {"id": s.id, "name": s.name, "type": s.type.value, "depends_on": s.depends_on}
                for s in wf.steps
            ],
        }

    @app.post("/v1/workflows/{workflow_id}/run", tags=["workflows"])
    async def run_workflow(
        workflow_id: str,
        variables: dict[str, Any] | None = None,
        _key=Depends(require_permission(Permission.PIPELINE_ASK)),
    ):
        try:
            run = await workflow_engine.execute(workflow_id, variables=variables)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        return {
            "run_id": run.id,
            "status": run.status.value,
            "steps_completed": sum(
                1 for s in run.step_results.values()
                if s.status.value == "completed"
            ),
            "error": run.error,
        }

    @app.get("/v1/workflows/{workflow_id}/runs", tags=["workflows"])
    async def list_workflow_runs(
        workflow_id: str,
        limit: int = 50,
        _key=Depends(get_current_key),
    ):
        return workflow_engine.list_runs(workflow_id=workflow_id, limit=limit)

    @app.get("/v1/runs/{run_id}", tags=["workflows"])
    async def get_run(run_id: str, _key=Depends(get_current_key)):
        run = workflow_engine.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return {
            "id": run.id,
            "workflow_id": run.workflow_id,
            "status": run.status.value,
            "current_step": run.current_step,
            "error": run.error,
            "started_at": run.started_at,
            "completed_at": run.completed_at,
            "steps": {
                sid: {
                    "status": sr.status.value,
                    "output": sr.output[:500],
                    "error": sr.error,
                    "duration_ms": sr.duration_ms,
                }
                for sid, sr in run.step_results.items()
            },
        }

    @app.post("/v1/runs/{run_id}/approve/{step_id}", tags=["workflows"])
    async def approve_step(
        run_id: str,
        step_id: str,
        approved: bool = True,
        _key=Depends(require_permission(Permission.ADMIN_CONFIG)),
    ):
        if not workflow_engine.approve_step(run_id, step_id, approved):
            raise HTTPException(status_code=404, detail="No pending approval found")
        return {"approved": approved}

    # ── WebSocket for streaming ────────────────────────────────────────

    @app.websocket("/v1/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        pipeline = _make_pipeline()

        try:
            while True:
                data = await websocket.receive_json()
                action = data.get("action", "ask")
                trace_id = data.get("trace_id", new_trace_id())
                set_trace_context(trace_id)

                try:
                    if action == "ask":
                        text = data.get("text", "")
                        llm = data.get("llm")
                        system_prompt = data.get("system_prompt")

                        await websocket.send_json({"type": "status", "message": "processing"})

                        response = await pipeline.ask(
                            text,
                            llm_name=llm,
                            output_name="json",
                            system_prompt=system_prompt,
                        )

                        await websocket.send_json({
                            "type": "response",
                            "text": response.text,
                            "model": response.model,
                            "usage": response.usage,
                            "trace_id": trace_id,
                        })

                    elif action == "transcribe":
                        audio_b64 = data.get("audio_base64", "")
                        fmt = data.get("format", "wav")
                        audio_bytes = base64.b64decode(audio_b64)
                        chunk = AudioChunk(data=audio_bytes, format=fmt)
                        text = await pipeline.transcribe_audio(chunk)
                        await websocket.send_json({
                            "type": "transcription",
                            "text": text,
                            "trace_id": trace_id,
                        })

                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Unknown action: {action}",
                        })

                except Exception as e:
                    logger.error(f"WebSocket error: {e}", exc_info=True)
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e),
                        "trace_id": trace_id,
                    })

        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")


def run_server(
    config: VaskConfig | None = None,
    host: str = "0.0.0.0",
    port: int = 8420,
) -> None:
    """Start the Vask API server."""
    import uvicorn

    app = create_app(config)
    logger.info(f"Starting Vask API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
