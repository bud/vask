"""HTTP/REST tool — make API calls from LLM tool invocations."""

from __future__ import annotations

from typing import Any

import httpx

from vask.config import ProviderConfig

BLOCKED_HOSTS = [
    "169.254.169.254",  # cloud metadata (AWS/GCP/Azure)
    "metadata.google.internal",
    "100.100.100.200",  # Alibaba metadata
]


class HttpTool:
    """Generic HTTP tool for REST API calls."""

    name = "http_request"
    description = "Make an HTTP request to a URL. Supports GET, POST, PUT, DELETE."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "DELETE"],
                "description": "HTTP method",
            },
            "url": {"type": "string", "description": "The URL to request"},
            "headers": {
                "type": "object",
                "description": "Optional headers",
                "additionalProperties": {"type": "string"},
            },
            "body": {"type": "string", "description": "Optional request body (JSON string)"},
        },
        "required": ["method", "url"],
    }

    def __init__(self, config: ProviderConfig | None = None) -> None:
        self._timeout = 30.0
        self._block_private = True
        if config:
            if "timeout" in config.extra:
                self._timeout = float(config.extra["timeout"])
            self._block_private = config.extra.get("block_private_networks", True)

    def _is_blocked_url(self, url: str) -> bool:
        from urllib.parse import urlparse

        parsed = urlparse(url)

        # Block non-HTTP schemes
        if parsed.scheme not in ("http", "https"):
            return True

        hostname = parsed.hostname or ""

        # Block cloud metadata endpoints
        if hostname in BLOCKED_HOSTS:
            return True

        if not self._block_private:
            return False

        # Block private/loopback IPs
        import ipaddress

        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                return True
        except ValueError:
            # Not an IP — check for localhost
            if hostname in ("localhost", "0.0.0.0"):
                return True

        return False

    async def execute(self, params: dict[str, Any]) -> str:
        method = params["method"]
        url = params["url"]
        headers = params.get("headers", {})
        body = params.get("body")

        if self._is_blocked_url(url):
            return f"Blocked: requests to private/internal networks are not allowed. URL: {url}"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.request(
                method=method,
                url=url,
                headers=headers,
                content=body,
            )
            return f"Status: {resp.status_code}\n{resp.text[:2000]}"
