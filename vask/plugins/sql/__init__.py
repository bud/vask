"""SQL connector plugin — read-only database tools for vask."""

from vask.plugins.sql.tools import SqlDescribeTable, SqlListTables, SqlQuery

TOOLS: list = [SqlQuery(), SqlListTables(), SqlDescribeTable()]

__all__ = ["TOOLS", "SqlQuery", "SqlListTables", "SqlDescribeTable"]
