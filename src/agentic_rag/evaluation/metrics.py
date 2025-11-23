from __future__ import annotations

import abc
from typing import Iterable, Sequence
from math import log2
from ..retrieval import Query, RetrievedChunk


class Metric(abc.ABC):
    name: str

    @abc.abstractmethod
    def compute(self, *, query: Query, retrieved: Sequence[RetrievedChunk], relevant: Iterable[str]) -> float:
        """Return the metric value for a single query."""


class MetricSuite:
    def __init__(self, metrics: Sequence[Metric]):
        self._metrics = metrics

    def evaluate(
        self,
        *,
        query: Query,
        retrieved: Sequence[RetrievedChunk],
        relevant: Iterable[str],
    ) -> dict[str, float]:
        return {metric.name: metric.compute(query=query, retrieved=retrieved, relevant=relevant) for metric in self._metrics}

class NDCG(Metric):
    def __init__(self, k: int = 10):
        self.k = k
        self.name = f"NDCG@{k}"

    def compute(self, *, query: Query, retrieved: Sequence[RetrievedChunk], relevant: Iterable[str]) -> float:
        if not relevant:
            return 0.0

        relevant_set = set(relevant)
        dcg = 0.0
        idcg = 0.0

        # DCG
        for i, chunk in enumerate(retrieved[: self.k]):
            if chunk.doc_id in relevant_set:
                dcg += 1 / log2(i + 2)  

        # IDCG (ideal)
        for i in range(min(len(relevant_set), self.k)):
            idcg += 1 / log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

class RecallAtK(Metric):
    def __init__(self, k: int = 10):
        self.k = k
        self.name = f"Recall@{k}"

    def compute(self, *, query: Query, retrieved: Sequence[RetrievedChunk], relevant: Iterable[str]) -> float:
        if not relevant:
            return 0.0
        relevant_set = set(relevant)
        retrieved_ids = {chunk.doc_id for chunk in retrieved[: self.k]}
        hits = len(retrieved_ids & relevant_set)
        return hits / len(relevant_set)

class PrecisionAtK(Metric):
    def __init__(self, k: int = 10):
        self.k = k
        self.name = f"Precision@{k}"

    def compute(self, *, query: Query, retrieved: Sequence[RetrievedChunk], relevant: Iterable[str]) -> float:
        if not retrieved:
            return 0.0
        relevant_set = set(relevant)
        retrieved_ids = {chunk.doc_id for chunk in retrieved[: self.k]}
        hits = len(retrieved_ids & relevant_set)
        return hits / len(retrieved_ids)

DEFAULT_METRICS = [
    RecallAtK(k=1),
    RecallAtK(k=5),
    RecallAtK(k=10),
    PrecisionAtK(k=10),    
    NDCG(k=10),
]