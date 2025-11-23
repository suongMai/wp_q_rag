from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(slots=True)
class Query:
    text: str
    metadata: Mapping[str, str] | None = None


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: Mapping[str, str] | None = None
