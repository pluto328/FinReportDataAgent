"""ES full-text retriever for text chunks."""

from __future__ import annotations

from app.core.retrieval.base import BaseRetriever
from app.infrastructure.es_client import ESClient
from app.schemas.document import ChunkStatus, DocChunk, ScoredChunk


class ESRetriever(BaseRetriever):
    name = "es"

    def __init__(self, es: ESClient) -> None:
        self._es = es

    async def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        hits = await self._es.search_chunks(query, top_k)
        return [self._to_scored(h) for h in hits]

    @staticmethod
    def _to_scored(hit: dict) -> ScoredChunk:
        chunk = DocChunk(
            doc_id=hit.get("doc_id", ""),
            chunk_id=hit.get("chunk_id", hit.get("id", "")),
            content=hit.get("content", ""),
            status=ChunkStatus(hit.get("status", "online")),
            source_file=hit.get("source_file", ""),
        )
        return ScoredChunk(chunk=chunk, score=float(hit.get("_score", 0.0)), retriever="es")
