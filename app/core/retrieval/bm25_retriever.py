"""BM25-style retriever via ES multi_match."""

from __future__ import annotations

from app.core.retrieval.es_retriever import ESRetriever


class BM25Retriever(ESRetriever):
    name = "bm25"

    async def search(self, query: str, top_k: int) -> list:
        hits = await self._es.search_chunks(query, top_k)
        results = [self._to_scored(h) for h in hits]
        for r in results:
            r.retriever = "bm25"
        return results
