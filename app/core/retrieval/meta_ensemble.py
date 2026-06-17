"""Structured metadata hybrid ensemble."""

from __future__ import annotations

import asyncio

from app.config.settings import Settings
from app.core.retrieval.base import BaseMetaRetriever
from app.core.retrieval.meta_dense_retriever import MetaDenseRetriever
from app.core.retrieval.meta_keyword_retriever import MetaKeywordRetriever
from app.core.retrieval.score_filter import filter_meta_by_min_score
from app.infrastructure.embedding_service import EmbeddingService
from app.schemas.structured import ScoredMetaRecord


class MetaEnsembleRetriever:
    def __init__(
        self,
        settings: Settings,
        keyword: MetaKeywordRetriever,
        dense: MetaDenseRetriever,
        embed: EmbeddingService,
    ) -> None:
        self._settings = settings
        self._embed = embed
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

        merged: dict[str, ScoredMetaRecord] = {}
        for name, batch in zip(enabled, batches):
            w = weights[name]
            for item in batch:
                key = item.record.asset_id
                score = item.score * w
                if key in merged:
                    merged[key].score += score
                else:
                    copy = item.model_copy(deep=True)
                    copy.score = score
                    merged[key] = copy
        merged_list = sorted(merged.values(), key=lambda x: x.score, reverse=True)[
            : self._settings.final_top_k
        ]
        return await filter_meta_by_min_score(
            self._embed,
            query,
            merged_list,
            self._settings.min_retrieval_score,
        )
