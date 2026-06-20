"""Cross-encoder reranker with optional score threshold filtering."""

from __future__ import annotations

from app.infrastructure.embedding_service import EmbeddingService
from app.schemas.document import ScoredChunk
from app.schemas.structured import ScoredMetaRecord


class Reranker:
    def __init__(self, embed: EmbeddingService) -> None:
        self._embed = embed

    async def rerank(
        self,
        query: str,
        items: list[ScoredChunk],
        top_k: int,
        *,
        min_score: float = 0.0,
    ) -> list[ScoredChunk]:
        if not items:
            return []
        docs = [i.chunk.content for i in items]
        ranked = await self._embed.rerank(query, docs, top_k=len(docs))
        out: list[ScoredChunk] = []
        for idx, score in ranked:
            score_f = float(score)
            if min_score > 0 and score_f < min_score:
                continue
            item = items[idx].model_copy(deep=True)
            item.score = score_f
            out.append(item)
            if len(out) >= top_k:
                break
        return out

    async def rerank_meta(
        self,
        query: str,
        items: list[ScoredMetaRecord],
        top_k: int,
        *,
        min_score: float = 0.0,
    ) -> list[ScoredMetaRecord]:
        if not items:
            return []
        docs = [i.record.search_text or i.record.file_name or "" for i in items]
        ranked = await self._embed.rerank(query, docs, top_k=len(docs))
        out: list[ScoredMetaRecord] = []
        for idx, score in ranked:
            score_f = float(score)
            if min_score > 0 and score_f < min_score:
                continue
            item = items[idx].model_copy(deep=True)
            item.score = score_f
            out.append(item)
            if len(out) >= top_k:
                break
        return out
