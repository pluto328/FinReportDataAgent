"""Text chunk hybrid ensemble: ES keyword + dense vector similarity."""

from __future__ import annotations

import asyncio

from app.config.settings import Settings
from app.core.retrieval.base import BaseRetriever
from app.core.retrieval.dense_retriever import DenseRetriever
from app.core.retrieval.es_retriever import ESRetriever
from app.core.retrieval.fusion import fuse_weighted_batches
from app.core.retrieval.reranker import Reranker
from app.schemas.document import ScoredChunk


class EnsembleRetriever:
    def __init__(
        self,
        settings: Settings,
        es: ESRetriever,
        dense: DenseRetriever,
        reranker: Reranker,
    ) -> None:
        self._settings = settings
        self._map: dict[str, BaseRetriever] = {
            "es": es,
            "dense": dense,
        }
        self._reranker = reranker

    async def search(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        k = top_k or self._settings.base_top_k
        enabled = self._settings.enabled_retrievers
        weights = self._settings.active_retrieval_weights()

        tasks = [self._map[name].search(query, k) for name in enabled]
        batches = await asyncio.gather(*tasks)
        channel_weights = [weights[name] for name in enabled]

        ranked = fuse_weighted_batches(
            list(batches),
            channel_weights,
            key_fn=lambda item: item.chunk.chunk_id,
            score_fn=lambda item: item.score,
            set_score=lambda item, s: item.model_copy(update={"score": s}),
            copy_fn=lambda item: item.model_copy(deep=True),
        )
        final_k = self._settings.final_top_k
        return await self._reranker.rerank(
            query,
            ranked[:k],
            final_k,
            min_score=self._settings.min_rerank_score,
        )
