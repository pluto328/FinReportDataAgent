"""Structured metadata hybrid ensemble: ES keyword + dense vector similarity."""

from __future__ import annotations

import asyncio

from app.config.settings import Settings
from app.core.retrieval.base import BaseMetaRetriever
from app.core.retrieval.fusion import fuse_weighted_batches
from app.core.retrieval.meta_dense_retriever import MetaDenseRetriever
from app.core.retrieval.meta_keyword_retriever import MetaKeywordRetriever
from app.core.retrieval.reranker import Reranker
from app.schemas.structured import ScoredMetaRecord


class MetaEnsembleRetriever:
    def __init__(
        self,
        settings: Settings,
        keyword: MetaKeywordRetriever,
        dense: MetaDenseRetriever,
        reranker: Reranker,
    ) -> None:
        self._settings = settings
        self._reranker = reranker
        self._map: dict[str, BaseMetaRetriever] = {
            "keyword": keyword,
            "dense": dense,
        }

    async def search(self, query: str, top_k: int | None = None) -> list[ScoredMetaRecord]:
        k = top_k or self._settings.base_top_k
        enabled = self._settings.enabled_meta_retrievers
        weights = self._settings.active_meta_retrieval_weights()
        tasks = [self._map[name].search(query, k) for name in enabled]
        batches = await asyncio.gather(*tasks)
        channel_weights = [weights[name] for name in enabled]

        merged_list = fuse_weighted_batches(
            list(batches),
            channel_weights,
            key_fn=lambda item: item.record.asset_id,
            score_fn=lambda item: item.score,
            set_score=lambda item, s: item.model_copy(update={"score": s}),
            copy_fn=lambda item: item.model_copy(deep=True),
        )[:k]
        return await self._reranker.rerank_meta(
            query,
            merged_list,
            self._settings.final_top_k,
            min_score=0.0,
        )
