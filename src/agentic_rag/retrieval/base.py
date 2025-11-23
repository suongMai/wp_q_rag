from __future__ import annotations

import abc
from typing import Iterable, Sequence
from .schemas import Query, RetrievedChunk


class BaseRetriever(abc.ABC):
    @abc.abstractmethod
    def search(self, query: Query, *, k: int = 5) -> Sequence[RetrievedChunk]:
        """Return the top-k retrieved chunks."""


class BaseReranker(abc.ABC):
    @abc.abstractmethod
    def rerank(self, query: Query, candidates: Iterable[RetrievedChunk], *, k: int = 5) -> Sequence[RetrievedChunk]:
        """Rerank retrieved candidates and return k best."""
