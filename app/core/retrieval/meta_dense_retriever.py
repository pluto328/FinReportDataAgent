"""Meta dense retriever via Chroma."""

from __future__ import annotations

from app.core.retrieval.base import BaseMetaRetriever
from app.infrastructure.embedding_service import EmbeddingService
from app.infrastructure.vector_client import VectorClient
from app.schemas.document import ChunkStatus
from app.schemas.structured import MetaRecord, ScoredMetaRecord, StructuredFormat


class MetaDenseRetriever(BaseMetaRetriever):
    name = "dense"

    def __init__(self, vectors: VectorClient, embed: EmbeddingService) -> None:
        self._vectors = vectors
        self._embed = embed

    async def search(self, query: str, top_k: int) -> list[ScoredMetaRecord]:
        qv = await self._embed.embed_query(query)
        hits = await self._vectors.query_meta(qv, top_k)
        results: list[ScoredMetaRecord] = []
        for h in hits:
            meta = h.get("metadata") or {}
            fmt = meta.get("format", "csv")
            try:
                structured_fmt = StructuredFormat(fmt)
            except ValueError:
                structured_fmt = StructuredFormat.CSV
            record = MetaRecord(
                asset_id=h.get("id", ""),
                file_path=meta.get("file_path", ""),
                file_name=meta.get("file_name", ""),
                format=structured_fmt,
                columns=meta.get("columns", "").split("|") if meta.get("columns") else [],
                search_text=h.get("document", ""),
                status=ChunkStatus(meta.get("status", "online")),
            )
            results.append(
                ScoredMetaRecord(
                    record=record,
                    score=float(h.get("_score", 0.0)),
                    retriever="dense",
                )
            )
        return results
