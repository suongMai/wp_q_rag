from __future__ import annotations

import abc
from typing import Sequence,List,Optional,Dict,Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from .types import Message, PlanStep,RetrievalStep
from ..retrieval.wordpress_retrieval import PgVectorRetriever,RetrievedChunk


class BaseAgentController(abc.ABC):
    """High-level orchestration contract."""

    @abc.abstractmethod
    def plan(self, history: Sequence[Message]) -> Sequence[PlanStep]:
        """Produce a plan (tool calls, retrieval steps, etc.) given the dialogue history."""

    @abc.abstractmethod
    def run(self, history: Sequence[Message]) -> Message:
        """Execute the plan and return the assistant's next message."""


class APIMessage(BaseModel):
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


class APIChatRequest(BaseModel):
    history: List[APIMessage]


class APIChatResponse(BaseModel):
    message: APIMessage
    context: Optional[str] = None
    retrieved_count: int = 0

class WordPressAgentController(BaseAgentController):

    def __init__(self, top_k: int=10):
        self.top_k = top_k
        self.retriver = PgVectorRetriever()


    def plan(self, history: Sequence[Message]) -> Sequence[PlanStep]:
        """
            
        """
        user_query = history[-1].content

        return [
            RetrievalStep(
                name="retrieve_wp_docs",
                query=user_query,
                top_k=self.top_k,
                rationale="Domain-specific WordPress question → retrieve from knowledge base"
            ),
            
        ]

    def run(self, history: Sequence[Message]) -> Message:
        """
            run the plan and return the result
        """
        user_query = history[-1].content
        plan = self.plan(history)

        retrieved_chunks: List[RetrievedChunk] = []
        context_str = "Could not find any related records."

        for step in plan:
            if isinstance(step, RetrievalStep):
                # 1) search returns list[str] -> wrap to RetrievedChunk
                chunks_raw = self.retriver.search(step.query, k=step.top_k)
                chunks_objs = [RetrievedChunk(text=c) if isinstance(c, str) else c for c in chunks_raw]
                retrieved_chunks.extend(chunks_objs)

                if chunks_objs:
                    context_str = "\n\n".join([
                        f"[{i+1}] {c.text[:500]}..."
                        for i, c in enumerate(chunks_objs[:5])
                    ])

        assistant_text = f"Answer based on retrieved context:\n{context_str}"

        return Message(
            role="assistant",
            content=assistant_text,
            metadata={"retrieved_count": len(retrieved_chunks)}
        )
        

    def serve(self, host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
        """
        Start a FastAPI server exposing /chat endpoint.
        For production consider Gunicorn + UvicornWorkers or containerization.
        """
        app = FastAPI(title="WordPress RAG Agent", debug=debug)

        @app.get("/health")
        def health():
            return {"status": "ok"}

        @app.post("/chat", response_model=APIChatResponse)
        def chat(req: APIChatRequest):
            try:
                if not req.history:
                    raise HTTPException(400, "`history` must be non-empty")

                
                internal_history = [
                    Message(role=m.role, content=m.content, metadata=m.metadata)
                    for m in req.history
                ]

                assistant_msg: Message = self.run(internal_history)

                
                api_msg = APIMessage(
                    role=assistant_msg.role,
                    content=assistant_msg.content,
                    metadata=getattr(assistant_msg, "metadata", None),
                )

                return APIChatResponse(
                    message=api_msg,
                    context=None,
                    retrieved_count=assistant_msg.metadata.get("retrieved_count", 0)
                )

            except Exception as exc:
                raise HTTPException(500, str(exc))

        uvicorn.run(app, host=host, port=port, log_level="info")