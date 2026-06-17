"""Dense vector retriever via Chroma."""

from __future__ import annotations

from app.core.retrieval.base import BaseRetriever
from app.infrastructure.embedding_service import EmbeddingService
from app.infrastructure.vector_client import VectorClient
from app.schemas.document import ChunkStatus, DocChunk, ScoredChunk


class DenseRetriever(BaseRetriever):
    name = "dense"

    def __init__(self, vectors: VectorClient, embed: EmbeddingService) -> None:
        self._vectors = vectors
        self._embed = embed

    async def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        qv = await self._embed.embed_query(query)
        hits = await self._vectors.query_chunks(qv, top_k)
        results: list[ScoredChunk] = []
        for h in hits:
            meta = h.get("metadata") or {}
            chunk = DocChunk(
                doc_id=meta.get("doc_id", ""),
                chunk_id=h.get("id", ""),
                content=h.get("document", ""),
                status=ChunkStatus(meta.get("status", "online")),
                source_file=meta.get("source_file", ""),
            )
            results.append(
                ScoredChunk(chunk=chunk, score=float(h.get("_score", 0.0)), retriever="dense")
            )
        return results
