"""Shell tool — execute local commands (opt-in only)."""

from __future__ import annotations

import asyncio
from typing import Any

from vask.config import ProviderConfig

BLOCKED_PATTERNS = [
    "rm -rf /", "rm -rf /*", "mkfs", "dd if=", "> /dev/sd",
    ":(){ :|:& };:", "chmod -R 777 /", "curl.*|.*sh", "wget.*|.*sh",
]


class ShellTool:
    """Execute shell commands locally. Must be explicitly enabled in config."""

    name = "shell_exec"
    description = "Execute a shell command and return its output. Use with caution."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to execute"},
        },
        "required": ["command"],
    }

    def __init__(self, config: ProviderConfig | None = None) -> None:
        self._enabled = False
        self._timeout = 30.0
        self._allowed_commands: list[str] = []  # empty = allow all (when enabled)
        if config:
            self._enabled = config.extra.get("enabled", False)
            self._timeout = float(config.extra.get("timeout", 30.0))
            self._allowed_commands = config.extra.get("allowed_commands", [])

    def _is_blocked(self, command: str) -> bool:
        cmd_lower = command.lower().strip()
        for pattern in BLOCKED_PATTERNS:
            if pattern in cmd_lower:
                return True
        return False

    async def execute(self, params: dict[str, Any]) -> str:
        if not self._enabled:
            return "Shell tool is disabled. Enable it in config with `enabled = true`."

        command = params["command"]

        if self._is_blocked(command):
            return "Command blocked: matches a dangerous pattern."

        if self._allowed_commands:
            base_cmd = command.split()[0] if command.split() else ""
            if base_cmd not in self._allowed_commands:
                return f"Command '{base_cmd}' not in allowed list: {self._allowed_commands}"

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self._timeout)
            output = stdout.decode() if stdout else ""
            errors = stderr.decode() if stderr else ""
            result = f"Exit code: {proc.returncode}\n"
            if output:
                result += f"stdout:\n{output[:2000]}\n"
            if errors:
                result += f"stderr:\n{errors[:1000]}"
            return result
        except TimeoutError:
            return f"Command timed out after {self._timeout}s"
