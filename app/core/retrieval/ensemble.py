"""Text chunk hybrid ensemble retriever."""

from __future__ import annotations

import asyncio

from app.config.settings import Settings
from app.core.retrieval.base import BaseRetriever
from app.core.retrieval.bm25_retriever import BM25Retriever
from app.core.retrieval.dense_retriever import DenseRetriever
from app.core.retrieval.es_retriever import ESRetriever
from app.core.retrieval.reranker import Reranker
from app.core.retrieval.score_filter import filter_chunks_by_min_score
from app.schemas.document import ScoredChunk


class EnsembleRetriever:
    def __init__(
        self,
        settings: Settings,
        es: ESRetriever,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        reranker: Reranker,
    ) -> None:
        self._settings = settings
        self._map: dict[str, BaseRetriever] = {
            "es": es,
            "bm25": bm25,
            "dense": dense,
        }
        self._reranker = reranker

    async def search(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        k = top_k or self._settings.base_top_k
        enabled = self._settings.enabled_retrievers
        weights = self._settings.active_retrieval_weights()

        tasks = [self._map[name].search(query, k) for name in enabled]
        batches = await asyncio.gather(*tasks)

        merged: dict[str, ScoredChunk] = {}
        for name, batch in zip(enabled, batches):
            w = weights[name]
            for item in batch:
                key = item.chunk.chunk_id
                score = item.score * w
                if key in merged:
                    merged[key].score += score
                else:
                    copy = item.model_copy(deep=True)
                    copy.score = score
                    merged[key] = copy

        ranked = sorted(merged.values(), key=lambda x: x.score, reverse=True)
        final_k = self._settings.final_top_k
        reranked = await self._reranker.rerank(query, ranked[:k], final_k)
        return await filter_chunks_by_min_score(
            self._reranker._embed,
            query,
            reranked,
            self._settings.min_retrieval_score,
        )
