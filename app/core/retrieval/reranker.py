"""Cross-encoder reranker."""

from __future__ import annotations

from app.infrastructure.embedding_service import EmbeddingService
from app.schemas.document import ScoredChunk


class Reranker:
    def __init__(self, embed: EmbeddingService) -> None:
        self._embed = embed

    async def rerank(
        self,
        query: str,
        items: list[ScoredChunk],
        top_k: int,
    ) -> list[ScoredChunk]:
        if not items:
            return []
        docs = [i.chunk.content for i in items]
        ranked = await self._embed.rerank(query, docs, top_k)
        out: list[ScoredChunk] = []
        for idx, score in ranked:
            item = items[idx].model_copy(deep=True)
            item.score = float(score)
            out.append(item)
        return out
