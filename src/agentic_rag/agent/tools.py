from __future__ import annotations

import abc
from typing import Any, Mapping


class BaseTool(abc.ABC):
    name: str
    description: str

    @abc.abstractmethod
    def run(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Execute the tool with a JSON-serialisable payload."""
