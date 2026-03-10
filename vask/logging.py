"""Structured logging and tracing for Vask."""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

# Context variables for request tracing
_trace_id: ContextVar[str] = ContextVar("trace_id", default="")
_span_id: ContextVar[str] = ContextVar("span_id", default="")
_user_id: ContextVar[str] = ContextVar("user_id", default="")


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def new_span_id() -> str:
    return uuid.uuid4().hex[:8]


def set_trace_context(trace_id: str, user_id: str = "") -> None:
    _trace_id.set(trace_id)
    _span_id.set(new_span_id())
    if user_id:
        _user_id.set(user_id)


class StructuredFormatter(logging.Formatter):
    """JSON-lines log formatter with trace context."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Add trace context if available
        trace_id = _trace_id.get("")
        if trace_id:
            log_entry["trace_id"] = trace_id
        span_id = _span_id.get("")
        if span_id:
            log_entry["span_id"] = span_id
        user_id = _user_id.get("")
        if user_id:
            log_entry["user_id"] = user_id

        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        # Add exception info
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
            }

        return json.dumps(log_entry, default=str)


class HumanFormatter(logging.Formatter):
    """Human-readable log formatter for development."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[31;1m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        trace = _trace_id.get("")
        trace_str = f" [{trace[:8]}]" if trace else ""
        msg = record.getMessage()
        return f"{color}{record.levelname:8s}{self.RESET}{trace_str} {record.name}: {msg}"


def setup_logging(
    level: str = "INFO",
    json_output: bool = False,
) -> None:
    """Configure Vask logging."""
    root = logging.getLogger("vask")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    if json_output:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(HumanFormatter())

    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a namespaced Vask logger."""
    return logging.getLogger(f"vask.{name}")


@dataclass(slots=True)
class Span:
    """Simple tracing span for performance measurement."""

    name: str
    trace_id: str = ""
    span_id: str = field(default_factory=new_span_id)
    parent_span_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def __enter__(self) -> Span:
        self.trace_id = _trace_id.get("") or new_trace_id()
        self.parent_span_id = _span_id.get("")
        _span_id.set(self.span_id)
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.end_time = time.perf_counter()
        _span_id.set(self.parent_span_id)
        logger = get_logger("tracing")
        logger.debug(
            "span completed",
            extra={
                "extra_fields": {
                    "span_name": self.name,
                    "span_id": self.span_id,
                    "duration_ms": round((self.end_time - self.start_time) * 1000, 2),
                    "attributes": self.attributes,
                }
            },
        )

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append({
            "name": name,
            "timestamp": time.perf_counter(),
            "attributes": attributes or {},
        })

    @property
    def duration_ms(self) -> float:
        if self.end_time and self.start_time:
            return round((self.end_time - self.start_time) * 1000, 2)
        return 0.0


@dataclass(slots=True)
class AuditLog:
    """Audit log entry for compliance tracking."""

    timestamp: float = field(default_factory=time.time)
    trace_id: str = ""
    user_id: str = ""
    action: str = ""
    resource: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    outcome: str = "success"

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "trace_id": self.trace_id or _trace_id.get(""),
            "user_id": self.user_id or _user_id.get(""),
            "action": self.action,
            "resource": self.resource,
            "details": self.details,
            "outcome": self.outcome,
        }


class AuditLogger:
    """Writes audit log entries for compliance."""

    def __init__(self) -> None:
        self._logger = get_logger("audit")
        self._entries: list[dict[str, Any]] = []

    def log(
        self,
        action: str,
        resource: str,
        details: dict[str, Any] | None = None,
        outcome: str = "success",
        user_id: str = "",
    ) -> None:
        entry = AuditLog(
            user_id=user_id,
            action=action,
            resource=resource,
            details=details or {},
            outcome=outcome,
        )
        record = entry.to_dict()
        self._entries.append(record)
        self._logger.info(
            f"{action} {resource} [{outcome}]",
            extra={"extra_fields": {"audit": record}},
        )

    def get_entries(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._entries[-limit:]


# Global audit logger
audit = AuditLogger()
