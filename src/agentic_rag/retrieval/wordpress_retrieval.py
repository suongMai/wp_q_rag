import json
import psycopg2
from pgvector.psycopg2 import register_vector
from typing import Sequence,Iterable
from sentence_transformers import SentenceTransformer, CrossEncoder
from .base import BaseRetriever, BaseReranker
from .schemas import Query, RetrievedChunk
from ..settings.schema import get_settings
class PgVectorRetriever(BaseRetriever):
    """Fast semantic search using pgvector + HNSW"""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        table_name: str = "wordpress_chunks",
        top_k: int = 20,  
    ):
        self.settings = get_settings()
        self.model = SentenceTransformer(model_name)
        self.table_name = table_name
        self.default_k = top_k

        self.conn  =psycopg2.connect(self.settings.database.url)
        register_vector(self.conn)

    def search(self, query: Query, *, k: int | None = None) -> Sequence[RetrievedChunk]:
        k = k or self.default_k
        embedding = self.model.encode(query, normalize_embeddings=True)

        sql = f"""
            SELECT id, doc_id, text, metadata, embedding <=> %s AS distance
            FROM {self.table_name}
            ORDER BY embedding <=> %s
            LIMIT %s
        """

        with self.conn.cursor() as cur:
            cur.execute(sql, (embedding, embedding, k))
            rows = cur.fetchall()

        results = []
        for row in rows:
            chunk_id, doc_id, text, metadata_json, distance = row
            metadata = json.loads(metadata_json) if metadata_json else {}
            results.append(RetrievedChunk(
                chunk_id=str(chunk_id),
                doc_id=doc_id or metadata.get("doc_id", ""),
                text=text,
                score=1.0 - float(distance),
                metadata=metadata
            ))

        return results
    


class CrossEncoderReranker(BaseReranker):
    """Second-stage reranker – kế thừa đúng BaseReranker"""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        default_k: int = 8
    ):
        self.model = CrossEncoder(model_name, max_length=512)
        self.default_k = default_k

    def rerank(
        self,
        query: Query,
        candidates: Iterable[RetrievedChunk],
        *,
        k: int | None = None
    ) -> Sequence[RetrievedChunk]:
        candidates = list(candidates)
        if not candidates:
            return []

        k = k or self.default_k
        if len(candidates) <= k:
            return candidates[:k]

        pairs = [(query, chunk.text) for chunk in candidates]
        scores = self.model.predict(pairs)

        for chunk, score in zip(candidates, scores):
            chunk.score = float(score)

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:k]