from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping


def _clean_nul(obj):
    if isinstance(obj, str):
        return obj.replace("\x00", "")
    elif isinstance(obj, dict):
        return {k: _clean_nul(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_nul(v) for v in obj]
    return obj

@dataclass(slots=True)
class RawRecord:    
    title: str    
    body: str| None
    identifier: str | None
    text: str |None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    def __post_init__(self):
        if self.title:
            self.title = self.title.replace("\x00", "")
        if self.body:
            self.body = self.body.replace("\x00", "")
        if self.identifier:
            self.identifier = self.identifier.replace("\x00", "")
        if self.text:
            self.text = self.text.replace("\x00", "")
        if self.metadata:
            self.metadata = _clean_nul(self.metadata)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    record_id: str
    text: str |None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
