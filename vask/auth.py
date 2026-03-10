"""Authentication and authorization for Vask API."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from vask.logging import audit, get_logger

logger = get_logger("auth")


class Permission(StrEnum):
    """Granular permissions for tool and resource access."""

    # Pipeline
    PIPELINE_RECORD = "pipeline:record"
    PIPELINE_ASK = "pipeline:ask"
    PIPELINE_TRANSCRIBE = "pipeline:transcribe"

    # Tools
    TOOLS_EXECUTE = "tools:execute"
    TOOLS_HTTP = "tools:http"
    TOOLS_SHELL = "tools:shell"

    # Plugins
    PLUGINS_SLACK = "plugins:slack"
    PLUGINS_GOOGLE = "plugins:google_workspace"
    PLUGINS_SQL = "plugins:sql"

    # Admin
    ADMIN_CONFIG = "admin:config"
    ADMIN_USERS = "admin:users"
    ADMIN_AUDIT = "admin:audit"


class Role(StrEnum):
    """Predefined roles with permission sets."""

    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"
    SERVICE = "service"


# Default role → permission mappings
ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.ADMIN: set(Permission),  # all permissions
    Role.USER: {
        Permission.PIPELINE_RECORD,
        Permission.PIPELINE_ASK,
        Permission.PIPELINE_TRANSCRIBE,
        Permission.TOOLS_EXECUTE,
        Permission.TOOLS_HTTP,
        Permission.PLUGINS_SLACK,
        Permission.PLUGINS_GOOGLE,
        Permission.PLUGINS_SQL,
    },
    Role.VIEWER: {
        Permission.PIPELINE_ASK,
        Permission.PIPELINE_TRANSCRIBE,
    },
    Role.SERVICE: {
        Permission.PIPELINE_RECORD,
        Permission.PIPELINE_ASK,
        Permission.PIPELINE_TRANSCRIBE,
        Permission.TOOLS_EXECUTE,
        Permission.TOOLS_HTTP,
        Permission.PLUGINS_SLACK,
        Permission.PLUGINS_GOOGLE,
        Permission.PLUGINS_SQL,
    },
}


@dataclass(slots=True)
class APIKey:
    """An API key with associated identity and permissions."""

    key_id: str
    key_hash: str  # SHA-256 hash of the key
    name: str
    role: Role = Role.USER
    permissions: set[Permission] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    rate_limit: int = 60  # requests per minute
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    @property
    def effective_permissions(self) -> set[Permission]:
        """Permissions from role + explicit grants."""
        role_perms = ROLE_PERMISSIONS.get(self.role, set())
        return role_perms | self.permissions


@dataclass(slots=True)
class AuthResult:
    """Result of an authentication attempt."""

    authenticated: bool
    key: APIKey | None = None
    error: str = ""


@dataclass(slots=True)
class RateLimitEntry:
    """Tracks request counts for rate limiting."""

    window_start: float = 0.0
    count: int = 0


class AuthManager:
    """Manages API keys, authentication, and authorization."""

    def __init__(self) -> None:
        self._keys: dict[str, APIKey] = {}  # key_id -> APIKey
        self._rate_limits: dict[str, RateLimitEntry] = {}
        self._load_from_env()

    def _load_from_env(self) -> None:
        """Load API keys from VASK_API_KEYS env var."""
        keys_raw = os.environ.get("VASK_API_KEYS", "")
        if not keys_raw:
            return
        # Format: "name1:key1:role1,name2:key2:role2"
        for entry in keys_raw.split(","):
            parts = entry.strip().split(":")
            if len(parts) >= 2:
                name = parts[0]
                raw_key = parts[1]
                role_str = parts[2] if len(parts) > 2 else "user"
                try:
                    role = Role(role_str)
                except ValueError:
                    role = Role.USER
                self.register_key(name, raw_key, role)

    @staticmethod
    def _hash_key(raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode()).hexdigest()

    @staticmethod
    def generate_key() -> str:
        """Generate a new API key."""
        return f"vask_{secrets.token_urlsafe(32)}"

    def register_key(
        self,
        name: str,
        raw_key: str,
        role: Role = Role.USER,
        permissions: set[Permission] | None = None,
        expires_at: float | None = None,
        rate_limit: int = 60,
    ) -> APIKey:
        """Register a new API key."""
        key_hash = self._hash_key(raw_key)
        key_id = key_hash[:12]
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            role=role,
            permissions=permissions or set(),
            expires_at=expires_at,
            rate_limit=rate_limit,
        )
        self._keys[key_id] = api_key
        logger.info(f"Registered API key '{name}' with role {role.value}")
        return api_key

    def authenticate(self, raw_key: str) -> AuthResult:
        """Authenticate a raw API key."""
        key_hash = self._hash_key(raw_key)
        key_id = key_hash[:12]

        api_key = self._keys.get(key_id)
        if not api_key:
            audit.log("auth", "api_key", {"reason": "unknown_key"}, outcome="denied")
            return AuthResult(authenticated=False, error="Invalid API key")

        if not api_key.enabled:
            details = {"key_name": api_key.name, "reason": "disabled"}
            audit.log("auth", "api_key", details, outcome="denied")
            return AuthResult(authenticated=False, error="API key is disabled")

        if api_key.is_expired:
            details = {"key_name": api_key.name, "reason": "expired"}
            audit.log("auth", "api_key", details, outcome="denied")
            return AuthResult(authenticated=False, error="API key has expired")

        if not hmac.compare_digest(api_key.key_hash, key_hash):
            audit.log("auth", "api_key", {"reason": "hash_mismatch"}, outcome="denied")
            return AuthResult(authenticated=False, error="Invalid API key")

        audit.log(
            "auth", "api_key", {"key_name": api_key.name},
            outcome="success", user_id=api_key.key_id,
        )
        return AuthResult(authenticated=True, key=api_key)

    def authorize(self, api_key: APIKey, permission: Permission) -> bool:
        """Check if an API key has a specific permission."""
        allowed = permission in api_key.effective_permissions
        if not allowed:
            audit.log(
                "authz",
                permission.value,
                {"key_name": api_key.name, "role": api_key.role.value},
                outcome="denied",
                user_id=api_key.key_id,
            )
        return allowed

    def check_rate_limit(self, api_key: APIKey) -> bool:
        """Check if the API key is within rate limits. Returns True if allowed."""
        now = time.time()
        entry = self._rate_limits.get(api_key.key_id)

        if entry is None or now - entry.window_start >= 60:
            self._rate_limits[api_key.key_id] = RateLimitEntry(window_start=now, count=1)
            return True

        if entry.count >= api_key.rate_limit:
            return False

        entry.count += 1
        return True

    def revoke_key(self, key_id: str) -> bool:
        """Disable an API key."""
        if key_id in self._keys:
            self._keys[key_id].enabled = False
            audit.log("auth", "revoke_key", {"key_id": key_id})
            return True
        return False

    def list_keys(self) -> list[dict[str, Any]]:
        """List all registered keys (without hashes)."""
        return [
            {
                "key_id": k.key_id,
                "name": k.name,
                "role": k.role.value,
                "enabled": k.enabled,
                "created_at": k.created_at,
                "expires_at": k.expires_at,
                "rate_limit": k.rate_limit,
            }
            for k in self._keys.values()
        ]
