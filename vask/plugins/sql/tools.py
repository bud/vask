"""SQL connector tools — read-only database access for the LLM."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import aiosqlite

# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

_WRITE_PATTERN = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)

_VALID_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")

DEFAULT_ROW_LIMIT = 100


def _parse_database_url(url: str) -> str:
    """Return a filesystem path from a DATABASE_URL value.

    Accepts ``sqlite:///path/to/db`` (strips the scheme) or a bare file path.
    """
    if url.startswith("sqlite:///"):
        return url[len("sqlite:///"):]
    if url.startswith("sqlite://"):
        return url[len("sqlite://"):]
    return url


def _resolve_connection(name: str = "default") -> str:
    """Return the database file path for *name*.

    Resolution order:
    1. ``VASK_SQL_CONNECTIONS`` env var — a JSON dict mapping names to URLs.
    2. ``DATABASE_URL`` env var (only used when *name* is ``"default"``).

    Raises ``ValueError`` when no matching connection is found.
    """
    connections_json = os.environ.get("VASK_SQL_CONNECTIONS")
    if connections_json:
        try:
            connections: dict[str, str] = json.loads(connections_json)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "VASK_SQL_CONNECTIONS is not valid JSON"
            ) from exc
        if name in connections:
            return _parse_database_url(connections[name])

    if name == "default":
        db_url = os.environ.get("DATABASE_URL")
        if db_url:
            return _parse_database_url(db_url)

    raise ValueError(
        f"No SQL connection configured for '{name}'. "
        "Set DATABASE_URL or VASK_SQL_CONNECTIONS."
    )


async def _get_connection(name: str = "default") -> aiosqlite.Connection:
    """Open (and return) an aiosqlite connection for *name*."""
    path = _resolve_connection(name)
    conn = await aiosqlite.connect(path)
    conn.row_factory = aiosqlite.Row
    return conn


def _validate_identifier(value: str, label: str = "identifier") -> None:
    """Raise ``ValueError`` if *value* is not a safe SQL identifier."""
    if not _VALID_IDENTIFIER.match(value):
        raise ValueError(
            f"Invalid {label}: {value!r}. "
            "Only alphanumeric characters, underscores, and dots are allowed."
        )


def _format_table(columns: list[str], rows: list[tuple[Any, ...]], total: int) -> str:
    """Return a human-readable text table with column headers."""
    if not rows:
        return "(0 rows)"

    # Compute column widths
    str_rows = [[str(v) for v in row] for row in rows]
    widths = [max(len(c), *(len(r[i]) for r in str_rows)) for i, c in enumerate(columns)]

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    header = "| " + " | ".join(c.ljust(w) for c, w in zip(columns, widths)) + " |"

    lines = [sep, header, sep]
    for row in str_rows:
        lines.append("| " + " | ".join(v.ljust(w) for v, w in zip(row, widths)) + " |")
    lines.append(sep)

    shown = len(rows)
    if shown < total:
        lines.append(f"({shown} of {total} rows shown — limit {DEFAULT_ROW_LIMIT})")
    else:
        lines.append(f"({shown} rows)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class SqlQuery:
    """Execute a read-only SQL query against a configured database."""

    name = "sql_query"
    description = (
        "Execute a read-only SQL SELECT query and return the results as a "
        "formatted table. Write operations (INSERT, UPDATE, DELETE, DROP, etc.) "
        "are rejected."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The SQL SELECT query to execute.",
            },
            "connection_name": {
                "type": "string",
                "description": "Named connection from VASK_SQL_CONNECTIONS (default: 'default').",
                "default": "default",
            },
        },
        "required": ["query"],
    }

    async def execute(self, params: dict[str, Any]) -> str:
        query: str = params.get("query", "").strip()
        connection_name: str = params.get("connection_name", "default")

        if not query:
            return "Error: query parameter is required."

        # Enforce read-only
        if _WRITE_PATTERN.match(query):
            return (
                "Error: write operations are not allowed. "
                "Only SELECT / read-only queries are permitted."
            )

        conn: aiosqlite.Connection | None = None
        try:
            conn = await _get_connection(connection_name)

            cursor = await conn.execute(query)
            all_rows = await cursor.fetchall()
            total = len(all_rows)
            rows = all_rows[:DEFAULT_ROW_LIMIT]

            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            return _format_table(columns, [tuple(r) for r in rows], total)

        except ValueError as exc:
            return f"Configuration error: {exc}"
        except Exception as exc:
            return f"SQL error: {exc}"
        finally:
            if conn:
                await conn.close()


class SqlDescribeTable:
    """Get the schema / structure of a specific database table."""

    name = "sql_describe_table"
    description = (
        "Return the column names, types, and constraints for a given table."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "table_name": {
                "type": "string",
                "description": "Name of the table to describe.",
            },
            "connection_name": {
                "type": "string",
                "description": "Named connection from VASK_SQL_CONNECTIONS (default: 'default').",
                "default": "default",
            },
        },
        "required": ["table_name"],
    }

    async def execute(self, params: dict[str, Any]) -> str:
        table_name: str = params.get("table_name", "").strip()
        connection_name: str = params.get("connection_name", "default")

        if not table_name:
            return "Error: table_name parameter is required."

        try:
            _validate_identifier(table_name, "table name")
        except ValueError as exc:
            return f"Error: {exc}"

        conn: aiosqlite.Connection | None = None
        try:
            conn = await _get_connection(connection_name)

            cursor = await conn.execute(f"PRAGMA table_info({table_name})")
            rows = await cursor.fetchall()

            if not rows:
                return f"Table '{table_name}' not found or has no columns."

            columns = ["cid", "name", "type", "notnull", "default_value", "pk"]
            return f"Table: {table_name}\n" + _format_table(
                columns, [tuple(r) for r in rows], len(rows)
            )

        except ValueError as exc:
            return f"Configuration error: {exc}"
        except Exception as exc:
            return f"SQL error: {exc}"
        finally:
            if conn:
                await conn.close()


class SqlListTables:
    """List all tables (and optionally filter by schema) in the database."""

    name = "sql_list_tables"
    description = "List all tables in the connected SQLite database."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "connection_name": {
                "type": "string",
                "description": "Named connection from VASK_SQL_CONNECTIONS (default: 'default').",
                "default": "default",
            },
            "schema": {
                "type": "string",
                "description": (
                    "Optional schema name to filter tables. "
                    "For SQLite this is typically 'main'."
                ),
            },
        },
        "required": [],
    }

    async def execute(self, params: dict[str, Any]) -> str:
        connection_name: str = params.get("connection_name", "default")
        schema: str | None = params.get("schema")

        if schema:
            try:
                _validate_identifier(schema, "schema")
            except ValueError as exc:
                return f"Error: {exc}"

        conn: aiosqlite.Connection | None = None
        try:
            conn = await _get_connection(connection_name)

            if schema:
                query = (
                    "SELECT name FROM pragma_table_list "
                    "WHERE schema = ? AND type = 'table' "
                    "ORDER BY name"
                )
                cursor = await conn.execute(query, (schema,))
            else:
                query = (
                    "SELECT name FROM sqlite_master "
                    "WHERE type = 'table' ORDER BY name"
                )
                cursor = await conn.execute(query)

            rows = await cursor.fetchall()

            if not rows:
                return "(0 tables found)"

            table_names = [row[0] for row in rows]
            result = f"Tables ({len(table_names)}):\n"
            for tname in table_names:
                result += f"  - {tname}\n"
            return result.rstrip()

        except ValueError as exc:
            return f"Configuration error: {exc}"
        except Exception as exc:
            return f"SQL error: {exc}"
        finally:
            if conn:
                await conn.close()
