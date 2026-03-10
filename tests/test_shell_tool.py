"""Tests for vask.tools.shell.ShellTool."""

import pytest

from vask.config import ProviderConfig
from vask.tools.shell import ShellTool


class TestShellToolDisabled:
    @pytest.mark.asyncio
    async def test_disabled_by_default_returns_message(self):
        tool = ShellTool()
        result = await tool.execute({"command": "echo hi"})
        assert "disabled" in result.lower()


class TestShellToolBlocking:
    def setup_method(self):
        self.tool = ShellTool(
            ProviderConfig(name="shell", type="tool", extra={"enabled": True})
        )

    @pytest.mark.parametrize(
        "cmd",
        ["rm -rf /", "rm -rf /*", "mkfs /dev/sda", "dd if=/dev/zero of=/dev/sda"],
    )
    @pytest.mark.asyncio
    async def test_blocks_dangerous_patterns(self, cmd):
        result = await self.tool.execute({"command": cmd})
        assert "blocked" in result.lower()

    def test_is_blocked_detects_patterns(self):
        assert self.tool._is_blocked("rm -rf /") is True
        assert self.tool._is_blocked("echo hello") is False


class TestShellToolAllowedCommands:
    @pytest.mark.asyncio
    async def test_allowed_commands_whitelist(self):
        tool = ShellTool(
            ProviderConfig(
                name="shell",
                type="tool",
                extra={"enabled": True, "allowed_commands": ["echo", "ls"]},
            )
        )
        result = await tool.execute({"command": "cat /etc/passwd"})
        assert "not in allowed list" in result

    @pytest.mark.asyncio
    async def test_allowed_command_passes(self):
        tool = ShellTool(
            ProviderConfig(
                name="shell",
                type="tool",
                extra={"enabled": True, "allowed_commands": ["echo"]},
            )
        )
        result = await tool.execute({"command": "echo test"})
        assert "test" in result


class TestShellToolExecution:
    @pytest.mark.asyncio
    async def test_enabled_tool_executes_simple_command(self):
        tool = ShellTool(
            ProviderConfig(name="shell", type="tool", extra={"enabled": True})
        )
        result = await tool.execute({"command": "echo hello_vask"})
        assert "hello_vask" in result
        assert "Exit code: 0" in result
