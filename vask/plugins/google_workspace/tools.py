"""Google Workspace tool providers for Gmail, Calendar, and Sheets."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx
import jwt


async def _get_auth_headers() -> dict[str, str]:
    """Build Authorization headers from env credentials.

    Supports two modes:
    1. GOOGLE_ACCESS_TOKEN  — a raw OAuth2 access token (simplest).
    2. GOOGLE_SERVICE_ACCOUNT_JSON — path to a service-account JSON key file.
       A short-lived access token is minted via a self-signed JWT.
    """
    access_token = os.environ.get("GOOGLE_ACCESS_TOKEN")
    if access_token:
        return {"Authorization": f"Bearer {access_token}"}

    sa_path = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not sa_path:
        raise RuntimeError(
            "Set GOOGLE_ACCESS_TOKEN or GOOGLE_SERVICE_ACCOUNT_JSON to authenticate "
            "with Google APIs."
        )

    with open(sa_path) as f:
        sa_info = json.load(f)

    now = int(time.time())
    payload = {
        "iss": sa_info["client_email"],
        "scope": (
            "https://www.googleapis.com/auth/gmail.send "
            "https://www.googleapis.com/auth/gmail.readonly "
            "https://www.googleapis.com/auth/calendar "
            "https://www.googleapis.com/auth/spreadsheets"
        ),
        "aud": "https://oauth2.googleapis.com/token",
        "iat": now,
        "exp": now + 3600,
    }
    signed_jwt = jwt.encode(payload, sa_info["private_key"], algorithm="RS256")

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "assertion": signed_jwt,
            },
        )
        resp.raise_for_status()
        token = resp.json()["access_token"]

    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Gmail
# ---------------------------------------------------------------------------

class GmailSendEmail:
    """Send an email via Gmail."""

    name: str = "gmail_send_email"
    description: str = "Send an email using Gmail."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "Recipient email address."},
            "subject": {"type": "string", "description": "Email subject line."},
            "body": {"type": "string", "description": "Plain-text email body."},
            "cc": {"type": "string", "description": "CC email address (optional)."},
            "bcc": {"type": "string", "description": "BCC email address (optional)."},
        },
        "required": ["to", "subject", "body"],
    }

    async def execute(self, params: dict[str, Any]) -> str:
        import base64
        from email.mime.text import MIMEText

        headers = await _get_auth_headers()

        msg = MIMEText(params["body"])
        msg["To"] = params["to"]
        msg["Subject"] = params["subject"]
        if params.get("cc"):
            msg["Cc"] = params["cc"]
        if params.get("bcc"):
            msg["Bcc"] = params["bcc"]

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
                headers=headers,
                json={"raw": raw},
            )
            resp.raise_for_status()
            data = resp.json()

        return json.dumps({"status": "sent", "message_id": data.get("id", "")})


class GmailSearchEmails:
    """Search emails in Gmail."""

    name: str = "gmail_search_emails"
    description: str = "Search emails in Gmail using a query string."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Gmail search query (same syntax as the Gmail search box).",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return.",
                "default": 10,
            },
        },
        "required": ["query"],
    }

    async def execute(self, params: dict[str, Any]) -> str:
        headers = await _get_auth_headers()
        max_results = params.get("max_results", 10)

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://gmail.googleapis.com/gmail/v1/users/me/messages",
                headers=headers,
                params={"q": params["query"], "maxResults": max_results},
            )
            resp.raise_for_status()
            message_ids = [m["id"] for m in resp.json().get("messages", [])]

            results: list[dict[str, Any]] = []
            for mid in message_ids:
                detail = await client.get(
                    f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{mid}",
                    headers=headers,
                    params={"format": "metadata", "metadataHeaders": "Subject,From,Date"},
                )
                detail.raise_for_status()
                payload = detail.json()
                header_map: dict[str, str] = {}
                for h in payload.get("payload", {}).get("headers", []):
                    header_map[h["name"]] = h["value"]
                results.append(
                    {
                        "id": mid,
                        "subject": header_map.get("Subject", ""),
                        "from": header_map.get("From", ""),
                        "date": header_map.get("Date", ""),
                        "snippet": payload.get("snippet", ""),
                    }
                )

        return json.dumps({"count": len(results), "messages": results})


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------

class CalendarListEvents:
    """List upcoming Google Calendar events."""

    name: str = "calendar_list_events"
    description: str = "List upcoming events from Google Calendar."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "calendar_id": {
                "type": "string",
                "description": "Calendar ID to query.",
                "default": "primary",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of events to return.",
                "default": 10,
            },
            "time_min": {
                "type": "string",
                "description": "Lower bound (ISO 8601) for event start time.",
            },
        },
        "required": [],
    }

    async def execute(self, params: dict[str, Any]) -> str:
        from datetime import UTC, datetime

        headers = await _get_auth_headers()
        calendar_id = params.get("calendar_id", "primary")
        max_results = params.get("max_results", 10)
        time_min = params.get("time_min") or datetime.now(UTC).isoformat()

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events",
                headers=headers,
                params={
                    "maxResults": max_results,
                    "timeMin": time_min,
                    "singleEvents": "true",
                    "orderBy": "startTime",
                },
            )
            resp.raise_for_status()
            items = resp.json().get("items", [])

        events = [
            {
                "id": ev.get("id", ""),
                "summary": ev.get("summary", "(no title)"),
                "start": ev.get("start", {}).get("dateTime", ev.get("start", {}).get("date", "")),
                "end": ev.get("end", {}).get("dateTime", ev.get("end", {}).get("date", "")),
                "location": ev.get("location", ""),
            }
            for ev in items
        ]

        return json.dumps({"count": len(events), "events": events})


class CalendarCreateEvent:
    """Create a Google Calendar event."""

    name: str = "calendar_create_event"
    description: str = "Create a new event on Google Calendar."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "Event title."},
            "start_time": {
                "type": "string",
                "description": "Event start time in ISO 8601 format.",
            },
            "end_time": {
                "type": "string",
                "description": "Event end time in ISO 8601 format.",
            },
            "description": {"type": "string", "description": "Event description (optional)."},
            "attendees": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of attendee email addresses (optional).",
            },
        },
        "required": ["summary", "start_time", "end_time"],
    }

    async def execute(self, params: dict[str, Any]) -> str:
        headers = await _get_auth_headers()

        event_body: dict[str, Any] = {
            "summary": params["summary"],
            "start": {"dateTime": params["start_time"]},
            "end": {"dateTime": params["end_time"]},
        }
        if params.get("description"):
            event_body["description"] = params["description"]
        if params.get("attendees"):
            event_body["attendees"] = [{"email": e} for e in params["attendees"]]

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://www.googleapis.com/calendar/v3/calendars/primary/events",
                headers=headers,
                json=event_body,
            )
            resp.raise_for_status()
            data = resp.json()

        return json.dumps(
            {
                "status": "created",
                "event_id": data.get("id", ""),
                "html_link": data.get("htmlLink", ""),
            }
        )


# ---------------------------------------------------------------------------
# Sheets
# ---------------------------------------------------------------------------

class SheetsReadRange:
    """Read data from a Google Spreadsheet."""

    name: str = "sheets_read_range"
    description: str = "Read a range of cells from a Google Spreadsheet."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "spreadsheet_id": {
                "type": "string",
                "description": "The ID of the spreadsheet.",
            },
            "range": {
                "type": "string",
                "description": "A1 notation range to read (e.g. 'Sheet1!A1:D10').",
            },
        },
        "required": ["spreadsheet_id", "range"],
    }

    async def execute(self, params: dict[str, Any]) -> str:
        headers = await _get_auth_headers()
        spreadsheet_id = params["spreadsheet_id"]
        cell_range = params["range"]

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}"
                f"/values/{cell_range}",
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

        values = data.get("values", [])
        return json.dumps({"range": data.get("range", cell_range), "values": values})


class SheetsAppendRow:
    """Append a row to a Google Spreadsheet."""

    name: str = "sheets_append_row"
    description: str = "Append a row of values to a Google Spreadsheet."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "spreadsheet_id": {
                "type": "string",
                "description": "The ID of the spreadsheet.",
            },
            "range": {
                "type": "string",
                "description": "A1 notation of the table to append to (e.g. 'Sheet1!A:E').",
            },
            "values": {
                "type": "array",
                "items": {},
                "description": "List of cell values for the new row.",
            },
        },
        "required": ["spreadsheet_id", "range", "values"],
    }

    async def execute(self, params: dict[str, Any]) -> str:
        headers = await _get_auth_headers()
        spreadsheet_id = params["spreadsheet_id"]
        cell_range = params["range"]

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}"
                f"/values/{cell_range}:append",
                headers=headers,
                params={"valueInputOption": "USER_ENTERED"},
                json={"values": [params["values"]]},
            )
            resp.raise_for_status()
            data = resp.json()

        updated = data.get("updates", {})
        return json.dumps(
            {
                "status": "appended",
                "updated_range": updated.get("updatedRange", ""),
                "updated_rows": updated.get("updatedRows", 0),
            }
        )


# ---------------------------------------------------------------------------
# Public list consumed by __init__.py
# ---------------------------------------------------------------------------

TOOLS: list[Any] = [
    GmailSendEmail(),
    GmailSearchEmails(),
    CalendarListEvents(),
    CalendarCreateEvent(),
    SheetsReadRange(),
    SheetsAppendRow(),
]
