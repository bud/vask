"""JSON output — prints structured JSON to stdout."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict

from vask.config import ProviderConfig
from vask.core.types import Response


class JsonOutput:
    """Render response as JSON to stdout."""

    def __init__(self, config: ProviderConfig | None = None) -> None:
        pass

    async def render(self, response: Response) -> None:
        data = asdict(response)
        json.dump(data, sys.stdout, indent=2)
        sys.stdout.write("\n")
        sys.stdout.flush()
