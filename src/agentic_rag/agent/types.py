from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence, Optional, Dict

from pydantic import BaseModel
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass(slots=True)
class Message:
    role: Role
    content: str
    metadata: Mapping[str, Any] | None = None


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    schema: Mapping[str, Any]
    runner: Callable[[Mapping[str, Any]], Mapping[str, Any]]


@dataclass(slots=True)
class PlanStep:
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    depends_on: Sequence[str] = field(default_factory=tuple)

@dataclass(slots=True)
class RetrievalStep(PlanStep):
    query: str = ""
    top_k: int = 10
    rationale: str = "Find a information from knowledge base"

class MessageIn(BaseModel):
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    history: List[MessageIn]

class ChatResponse(BaseModel):
    message: MessageIn
    context: Optional[str] = None
    retrieved_count: int = 0