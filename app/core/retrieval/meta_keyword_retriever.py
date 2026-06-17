"""Meta keyword retriever via ES."""

from __future__ import annotations

from app.core.retrieval.base import BaseMetaRetriever
from app.infrastructure.es_client import ESClient
from app.schemas.document import ChunkStatus
from app.schemas.structured import MetaRecord, ScoredMetaRecord, StructuredFormat


class MetaKeywordRetriever(BaseMetaRetriever):
    name = "keyword"

    def __init__(self, es: ESClient) -> None:
        self._es = es

    async def search(self, query: str, top_k: int) -> list[ScoredMetaRecord]:
        hits = await self._es.search_meta_keyword(query, top_k)
        return [self._to_scored(h) for h in hits]

    @staticmethod
    def _to_scored(hit: dict) -> ScoredMetaRecord:
        fmt = hit.get("format", "csv")
        try:
            structured_fmt = StructuredFormat(fmt)
        except ValueError:
            structured_fmt = StructuredFormat.CSV
        record = MetaRecord(
            asset_id=hit.get("asset_id", ""),
            file_path=hit.get("file_path", ""),
            file_name=hit.get("file_name", ""),
            format=structured_fmt,
            columns=hit.get("columns", []) if isinstance(hit.get("columns"), list) else [],
            search_text=hit.get("search_text", ""),
            status=ChunkStatus(hit.get("status", "online")),
        )
        return ScoredMetaRecord(record=record, score=float(hit.get("_score", 0.0)), retriever="keyword")
