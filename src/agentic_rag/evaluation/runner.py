from __future__ import annotations

import abc
import json
from typing import Iterable, Dict, List
from pathlib import Path
from .metrics import DEFAULT_METRICS, MetricSuite
from ..agent.controller import WordPressAgentController,RetrievedChunk
from ..retrieval import Query


class BaseEvaluator(abc.ABC):
    @abc.abstractmethod
    def iter_queries(self) -> Iterable[Query]:
        """Yield evaluation queries."""

    @abc.abstractmethod
    def evaluate(self) -> None:
        """Run the evaluation suite."""

class WordPressEvaluator(BaseEvaluator):
    """
        Evaluator
    """

    def __init__(
        self,
        raw_dir: Path = Path("data/raw"),
        top_k: int = 10,
    ):
        self.raw_dir = raw_dir
        self.top_k = top_k
        self.controller = WordPressAgentController()                  
        self.suite = MetricSuite(DEFAULT_METRICS)

        self.qrels = self._load_qrels()      
        self.queries = self._load_queries()  

    def _load_qrels(self) -> Dict[str, List[str]]:
        qrels: Dict[str, List[str]] = {}
        path = self.raw_dir / "qrels.jsonl"
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("score", 0) >= 1:
                    qrels.setdefault(data["query-id"], []).append(data["corpus-id"])
        return qrels

    def _load_queries(self) -> Dict[str, str]:
        queries: Dict[str, str] = {}
        path = self.raw_dir / "queries.jsonl"
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                queries[data["_id"]] = data["text"]
        return queries
    def iter_queries(self) -> Iterable[Query]:
        for qid,text in self.queries.items():
            yield Query(text=text, metadata={"id": qid})


    def evaluate(self) -> None:
        total = len(self.queries)
        results: List[Dict[str, float]] = []

        print(f"Running evaluation on {total} queries (top_k={self.top_k})...")

        for i, query in enumerate(self.iter_queries(), 1):
            retrieved: List[RetrievedChunk] = self.controller.retriver.search(
                query.text, k=self.top_k
            )
            relevant = self.qrels.get(query.metadata.get("id"), [])

            scores = self.suite.evaluate(
                query=query,
                retrieved=retrieved,
                relevant=relevant,
            )
            results.append(scores)

            if i % 50 == 0 or i == total:
                print(f"Processed: {i}/{total}")

        avg = {}
        for key in results[0].keys():
            avg[key] = sum(r[key] for r in results) / len(results)

        print("\n" + "=" * 55)
        print("FINAL EVALUATION RESULTS – CQADupStack WordPress")
        print("=" * 55)
        for name, score in avg.items():
            print(f"{name:15} │ {score:.4f}")
        print("=" * 55)


if __name__ == "__main__":
    WordPressEvaluator().evaluate()