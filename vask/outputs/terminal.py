"""Rich terminal output renderer."""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown

from vask.config import ProviderConfig
from vask.core.types import Response


class TerminalOutput:
    """Render responses to the terminal with Rich formatting."""

    def __init__(self, config: ProviderConfig | None = None) -> None:
        self._console = Console()

    async def render(self, response: Response) -> None:
        if response.text:
            self._console.print(Markdown(response.text))
        if response.usage:
            tokens = response.usage.get("prompt_tokens", 0) + response.usage.get("completion_tokens", 0)
            self._console.print(
                f"\n[dim]model={response.model}  tokens={tokens}[/dim]",
            )
