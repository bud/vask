"""Webhook system — inbound triggers for Vask workflows."""

from __future__ import annotations

import hashlib
import hmac
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from vask.logging import audit, get_logger

logger = get_logger("webhooks")


class WebhookSource(StrEnum):
    """Known webhook sources with signature verification."""

    GENERIC = "generic"
    SLACK = "slack"
    GITHUB = "github"
    STRIPE = "stripe"
    TWILIO = "twilio"
    CUSTOM = "custom"


@dataclass(slots=True)
class WebhookConfig:
    """Configuration for a registered webhook endpoint."""

    id: str
    name: str
    source: WebhookSource = WebhookSource.GENERIC
    secret: str = ""  # For HMAC signature verification
    enabled: bool = True
    action: str = "ask"  # Pipeline action: ask, record, workflow
    action_config: dict[str, Any] = field(default_factory=dict)
    # Action config examples:
    #   {"llm": "claude", "system_prompt": "You are a Slack bot..."}
    #   {"workflow_id": "crm-update"}
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WebhookEvent:
    """An inbound webhook event."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    webhook_id: str = ""
    source: WebhookSource = WebhookSource.GENERIC
    timestamp: float = field(default_factory=time.time)
    headers: dict[str, str] = field(default_factory=dict)
    body: dict[str, Any] | str = field(default_factory=dict)
    raw_body: bytes = b""
    processed: bool = False
    result: str = ""
    error: str = ""


class WebhookRegistry:
    """Manages webhook endpoints and dispatches events."""

    def __init__(self) -> None:
        self._webhooks: dict[str, WebhookConfig] = {}
        self._handlers: dict[str, Callable[[WebhookEvent], Awaitable[str]]] = {}
        self._event_log: list[WebhookEvent] = []

    def register(
        self,
        name: str,
        source: WebhookSource = WebhookSource.GENERIC,
        secret: str = "",
        action: str = "ask",
        action_config: dict[str, Any] | None = None,
    ) -> WebhookConfig:
        """Register a new webhook endpoint."""
        webhook_id = uuid.uuid4().hex[:12]
        config = WebhookConfig(
            id=webhook_id,
            name=name,
            source=source,
            secret=secret,
            action=action,
            action_config=action_config or {},
        )
        self._webhooks[webhook_id] = config
        logger.info(f"Registered webhook '{name}' (id={webhook_id}, source={source.value})")
        return config

    def unregister(self, webhook_id: str) -> bool:
        return self._webhooks.pop(webhook_id, None) is not None

    def get(self, webhook_id: str) -> WebhookConfig | None:
        return self._webhooks.get(webhook_id)

    def list_webhooks(self) -> list[dict[str, Any]]:
        return [
            {
                "id": w.id,
                "name": w.name,
                "source": w.source.value,
                "enabled": w.enabled,
                "action": w.action,
                "created_at": w.created_at,
            }
            for w in self._webhooks.values()
        ]

    def set_handler(
        self,
        action: str,
        handler: Callable[[WebhookEvent], Awaitable[str]],
    ) -> None:
        """Register a handler function for an action type."""
        self._handlers[action] = handler

    def verify_signature(
        self,
        webhook: WebhookConfig,
        raw_body: bytes,
        headers: dict[str, str],
    ) -> bool:
        """Verify webhook signature based on source type."""
        if not webhook.secret:
            return True  # No secret configured, skip verification

        if webhook.source == WebhookSource.SLACK:
            return self._verify_slack(webhook.secret, raw_body, headers)
        elif webhook.source == WebhookSource.GITHUB:
            return self._verify_github(webhook.secret, raw_body, headers)
        elif webhook.source == WebhookSource.STRIPE:
            return self._verify_stripe(webhook.secret, raw_body, headers)
        else:
            return self._verify_hmac(webhook.secret, raw_body, headers)

    def _verify_hmac(
        self, secret: str, body: bytes, headers: dict[str, str]
    ) -> bool:
        """Generic HMAC-SHA256 verification."""
        signature = headers.get("x-signature-256", headers.get("x-hub-signature-256", ""))
        if not signature:
            return False
        expected = "sha256=" + hmac.new(
            secret.encode(), body, hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(signature, expected)

    def _verify_slack(
        self, secret: str, body: bytes, headers: dict[str, str]
    ) -> bool:
        """Slack request signing verification."""
        timestamp = headers.get("x-slack-request-timestamp", "")
        signature = headers.get("x-slack-signature", "")
        if not timestamp or not signature:
            return False
        # Check timestamp freshness (5 min window)
        try:
            if abs(time.time() - float(timestamp)) > 300:
                return False
        except ValueError:
            return False
        sig_basestring = f"v0:{timestamp}:{body.decode()}"
        computed = "v0=" + hmac.new(
            secret.encode(), sig_basestring.encode(), hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(computed, signature)

    def _verify_github(
        self, secret: str, body: bytes, headers: dict[str, str]
    ) -> bool:
        """GitHub webhook signature verification."""
        signature = headers.get("x-hub-signature-256", "")
        if not signature:
            return False
        expected = "sha256=" + hmac.new(
            secret.encode(), body, hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(signature, expected)

    def _verify_stripe(
        self, secret: str, body: bytes, headers: dict[str, str]
    ) -> bool:
        """Stripe webhook signature verification."""
        sig_header = headers.get("stripe-signature", "")
        if not sig_header:
            return False
        # Parse Stripe signature header
        elements: dict[str, str] = {}
        for item in sig_header.split(","):
            key, _, value = item.strip().partition("=")
            elements[key] = value
        timestamp = elements.get("t", "")
        v1_sig = elements.get("v1", "")
        if not timestamp or not v1_sig:
            return False
        signed_payload = f"{timestamp}.{body.decode()}"
        expected = hmac.new(
            secret.encode(), signed_payload.encode(), hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, v1_sig)

    async def dispatch(self, event: WebhookEvent) -> str:
        """Dispatch a webhook event to its handler."""
        webhook = self._webhooks.get(event.webhook_id)
        if not webhook:
            event.error = f"Unknown webhook: {event.webhook_id}"
            self._event_log.append(event)
            return event.error

        if not webhook.enabled:
            event.error = "Webhook is disabled"
            self._event_log.append(event)
            return event.error

        handler = self._handlers.get(webhook.action)
        if not handler:
            event.error = f"No handler for action: {webhook.action}"
            self._event_log.append(event)
            return event.error

        try:
            result = await handler(event)
            event.processed = True
            event.result = result
            audit.log("webhook", webhook.name, {
                "event_id": event.id,
                "source": webhook.source.value,
            })
        except Exception as e:
            event.error = str(e)
            logger.error(f"Webhook handler error: {e}", exc_info=True)
            audit.log("webhook", webhook.name, {
                "event_id": event.id,
                "error": str(e),
            }, outcome="error")

        self._event_log.append(event)
        return event.result or event.error

    def get_event_log(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent webhook events."""
        return [
            {
                "id": e.id,
                "webhook_id": e.webhook_id,
                "source": e.source.value,
                "timestamp": e.timestamp,
                "processed": e.processed,
                "result_preview": e.result[:200] if e.result else "",
                "error": e.error,
            }
            for e in self._event_log[-limit:]
        ]


def extract_text_from_webhook(source: WebhookSource, body: dict[str, Any]) -> str:
    """Extract the actionable text content from a webhook payload."""
    if source == WebhookSource.SLACK:
        # Slack event API
        event = body.get("event", {})
        return event.get("text", body.get("text", ""))
    elif source == WebhookSource.GITHUB:
        # GitHub issue/PR/comment
        action = body.get("action", "")
        if "comment" in body:
            return f"[GitHub {action}] {body['comment'].get('body', '')}"
        if "issue" in body:
            issue = body["issue"]
            return f"[GitHub issue {action}] {issue.get('title', '')}: {issue.get('body', '')}"
        if "pull_request" in body:
            pr = body["pull_request"]
            return f"[GitHub PR {action}] {pr.get('title', '')}: {pr.get('body', '')}"
        return str(body)
    elif source == WebhookSource.STRIPE:
        event_type = body.get("type", "")
        data = body.get("data", {}).get("object", {})
        return f"[Stripe {event_type}] {data}"
    elif source == WebhookSource.TWILIO:
        return body.get("Body", body.get("TranscriptionText", ""))
    else:
        # Generic: look for common text fields
        for key in ("text", "message", "body", "content", "query"):
            if key in body and isinstance(body[key], str):
                return body[key]
        return str(body)
