"""Slack tools — interact with Slack via the Web API."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

SLACK_API_BASE = "https://slack.com/api"
_TIMEOUT = 15.0


def _get_token() -> str:
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    if not token:
        raise RuntimeError("SLACK_BOT_TOKEN environment variable is not set")
    return token


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
    }


async def _slack_request(
    method: str,
    endpoint: str,
    *,
    payload: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Make an authenticated request to the Slack Web API."""
    token = _get_token()
    url = f"{SLACK_API_BASE}/{endpoint}"

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        if method == "GET":
            resp = await client.get(url, headers=_headers(token), params=params)
        else:
            resp = await client.post(url, headers=_headers(token), json=payload)

    data = resp.json()
    if not data.get("ok"):
        error = data.get("error", "unknown_error")
        raise RuntimeError(f"Slack API error: {error}")
    return data


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class SlackSendMessage:
    """Send a message to a Slack channel or user."""

    name = "slack_send_message"
    description = "Send a message to a Slack channel or user via the Slack Web API."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "channel": {
                "type": "string",
                "description": "Channel ID or user ID to send the message to",
            },
            "text": {
                "type": "string",
                "description": "Message text to send",
            },
            "thread_ts": {
                "type": "string",
                "description": "Optional thread timestamp to reply in a thread",
            },
        },
        "required": ["channel", "text"],
    }

    async def execute(self, params: dict[str, Any]) -> str:
        payload: dict[str, Any] = {
            "channel": params["channel"],
            "text": params["text"],
        }
        if "thread_ts" in params:
            payload["thread_ts"] = params["thread_ts"]

        try:
            data = await _slack_request("POST", "chat.postMessage", payload=payload)
            ts = data.get("ts", "")
            channel = data.get("channel", params["channel"])
            return f"Message sent to {channel} (ts={ts})"
        except RuntimeError as exc:
            return f"Failed to send message: {exc}"


class SlackListChannels:
    """List available Slack channels."""

    name = "slack_list_channels"
    description = "List available Slack channels the bot has access to."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Maximum number of channels to return",
                "default": 100,
            },
            "types": {
                "type": "string",
                "description": "Comma-separated channel types (e.g. public_channel, private_channel)",
                "default": "public_channel",
            },
        },
        "required": [],
    }

    async def execute(self, params: dict[str, Any]) -> str:
        query: dict[str, Any] = {
            "limit": params.get("limit", 100),
            "types": params.get("types", "public_channel"),
        }

        try:
            data = await _slack_request("GET", "conversations.list", params=query)
            channels = data.get("channels", [])
            if not channels:
                return "No channels found."
            lines = [f"Found {len(channels)} channel(s):"]
            for ch in channels:
                name = ch.get("name", ch.get("id", "?"))
                cid = ch.get("id", "?")
                purpose = ch.get("purpose", {}).get("value", "")
                entry = f"  #{name} ({cid})"
                if purpose:
                    entry += f" — {purpose}"
                lines.append(entry)
            return "\n".join(lines)
        except RuntimeError as exc:
            return f"Failed to list channels: {exc}"


class SlackSearchMessages:
    """Search Slack messages."""

    name = "slack_search_messages"
    description = "Search for messages across Slack using a query string."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string",
            },
            "count": {
                "type": "integer",
                "description": "Number of results to return",
                "default": 20,
            },
        },
        "required": ["query"],
    }

    async def execute(self, params: dict[str, Any]) -> str:
        query: dict[str, Any] = {
            "query": params["query"],
            "count": params.get("count", 20),
        }

        try:
            data = await _slack_request("GET", "search.messages", params=query)
            messages = data.get("messages", {})
            matches = messages.get("matches", [])
            total = messages.get("total", 0)
            if not matches:
                return f"No messages found for query: {params['query']}"
            lines = [f"Found {total} result(s) (showing {len(matches)}):"]
            for m in matches:
                user = m.get("username", m.get("user", "?"))
                text = m.get("text", "")[:200]
                channel_name = m.get("channel", {}).get("name", "?")
                ts = m.get("ts", "")
                lines.append(f"  [{channel_name}] {user} ({ts}): {text}")
            return "\n".join(lines)
        except RuntimeError as exc:
            return f"Failed to search messages: {exc}"


class SlackSetStatus:
    """Set the authenticated user's Slack status."""

    name = "slack_set_status"
    description = "Set the current user's status text and emoji in Slack."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "status_text": {
                "type": "string",
                "description": "Status text to display",
            },
            "status_emoji": {
                "type": "string",
                "description": "Status emoji (e.g. :house_with_garden:)",
            },
        },
        "required": ["status_text"],
    }

    async def execute(self, params: dict[str, Any]) -> str:
        profile: dict[str, Any] = {
            "status_text": params["status_text"],
            "status_emoji": params.get("status_emoji", ""),
        }
        payload = {"profile": json.dumps(profile)}

        try:
            await _slack_request("POST", "users.profile.set", payload=payload)
            emoji = params.get("status_emoji", "")
            display = f"{emoji} {params['status_text']}" if emoji else params["status_text"]
            return f"Status set to: {display}"
        except RuntimeError as exc:
            return f"Failed to set status: {exc}"


class SlackReact:
    """Add a reaction to a Slack message."""

    name = "slack_react"
    description = "Add an emoji reaction to a specific Slack message."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "channel": {
                "type": "string",
                "description": "Channel ID where the message lives",
            },
            "timestamp": {
                "type": "string",
                "description": "Timestamp of the message to react to",
            },
            "emoji": {
                "type": "string",
                "description": "Emoji name without colons (e.g. thumbsup)",
            },
        },
        "required": ["channel", "timestamp", "emoji"],
    }

    async def execute(self, params: dict[str, Any]) -> str:
        payload = {
            "channel": params["channel"],
            "timestamp": params["timestamp"],
            "name": params["emoji"],
        }

        try:
            await _slack_request("POST", "reactions.add", payload=payload)
            return f"Reacted with :{params['emoji']}: to message {params['timestamp']} in {params['channel']}"
        except RuntimeError as exc:
            return f"Failed to add reaction: {exc}"
