"""Elasticsearch client wrapper."""

from __future__ import annotations

from typing import Any

from elasticsearch import AsyncElasticsearch, NotFoundError
from elasticsearch.helpers import async_bulk

from app.config.settings import Settings


CHUNK_INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "doc_id": {"type": "keyword"},
            "chunk_id": {"type": "keyword"},
            "content": {"type": "text"},
            "status": {"type": "keyword"},
            "source_file": {"type": "keyword"},
            "asset_kind": {"type": "keyword"},
        }
    }
}

META_INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "asset_id": {"type": "keyword"},
            "file_path": {"type": "keyword"},
            "file_name": {"type": "text"},
            "search_text": {"type": "text"},
            "columns": {"type": "keyword"},
            "categories": {"type": "keyword"},
            "status": {"type": "keyword"},
            "format": {"type": "keyword"},
        }
    }
}


class ESClient:
    """ES 8.x async client with dual index support."""

    def __init__(self, settings: Settings) -> None:
        self._chunk_index = settings.es_index_name
        self._meta_index = settings.es_meta_index_name
        self._client = AsyncElasticsearch(
            settings.es_host,
            basic_auth=(settings.es_user, settings.es_password),
            request_timeout=60,
        )

    async def close(self) -> None:
        await self._client.close()

    async def ensure_indices(self) -> None:
        for name, body in (
            (self._chunk_index, CHUNK_INDEX_MAPPING),
            (self._meta_index, META_INDEX_MAPPING),
        ):
            if not await self._client.indices.exists(index=name):
                await self._client.indices.create(index=name, body=body)

    async def index_chunk(self, doc: dict[str, Any]) -> None:
        await self._client.index(index=self._chunk_index, id=doc["chunk_id"], document=doc)

    async def index_meta(self, doc: dict[str, Any]) -> None:
        await self._client.index(index=self._meta_index, id=doc["asset_id"], document=doc)

    async def bulk_index_chunks(self, docs: list[dict[str, Any]]) -> None:
        if not docs:
            return
        actions = (
            {"_index": self._chunk_index, "_id": doc["chunk_id"], "_source": doc} for doc in docs
        )
        await async_bulk(self._client, actions, raise_on_error=True)

    async def bulk_index_meta(self, docs: list[dict[str, Any]]) -> None:
        if not docs:
            return
        actions = (
            {"_index": self._meta_index, "_id": doc["asset_id"], "_source": doc} for doc in docs
        )
        await async_bulk(self._client, actions, raise_on_error=True)

    async def mark_chunk_invalid(self, doc_id: str) -> None:
        await self._client.update_by_query(
            index=self._chunk_index,
            body={
                "script": {"source": "ctx._source.status = 'invalid'", "lang": "painless"},
                "query": {"term": {"doc_id": doc_id}},
            },
        )

    async def mark_meta_invalid(self, asset_id: str) -> None:
        try:
            await self._client.update(
                index=self._meta_index,
                id=asset_id,
                doc={"status": "invalid"},
            )
        except NotFoundError:
            return

    async def search_chunks(self, query: str, top_k: int) -> list[dict[str, Any]]:
        resp = await self._client.search(
            index=self._chunk_index,
            body={
                "size": top_k,
                "query": {
                    "bool": {
                        "must": [{"match": {"content": query}}],
                        "filter": [{"term": {"status": "online"}}],
                    }
                },
            },
        )
        return [self._hit(h) for h in resp["hits"]["hits"]]

    async def search_meta_keyword(self, query: str, top_k: int) -> list[dict[str, Any]]:
        resp = await self._client.search(
            index=self._meta_index,
            body={
                "size": top_k,
                "query": {
                    "bool": {
                        "must": [{"multi_match": {"query": query, "fields": ["search_text", "file_name", "columns"]}}],
                        "filter": [{"term": {"status": "online"}}],
                    }
                },
            },
        )
        return [self._hit(h) for h in resp["hits"]["hits"]]

    @staticmethod
    def _hit(hit: dict[str, Any]) -> dict[str, Any]:
        src = hit["_source"]
        src["_score"] = hit["_score"]
        return src
