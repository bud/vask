"""Clipboard output — copies response text to the system clipboard."""

from __future__ import annotations

import subprocess

from vask.config import ProviderConfig
from vask.core.types import Response


class ClipboardOutput:
    """Copy response text to the system clipboard."""

    def __init__(self, config: ProviderConfig | None = None) -> None:
        pass

    async def render(self, response: Response) -> None:
        if response.text:
            subprocess.run(["pbcopy"], input=response.text.encode(), check=True)
