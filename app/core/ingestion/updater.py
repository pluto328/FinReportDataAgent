"""Incremental document sync with MD5 and soft-invalid."""

from __future__ import annotations

import asyncio
import gc
import json
import time
from pathlib import Path

from app.common.logger import logger
from app.config.settings import Settings, get_settings
from app.core.ingestion.chunker import Chunker
from app.core.ingestion.meta_extractor import STRUCTURED_EXTENSIONS, MetadataExtractor
from app.core.ingestion.parser import TEXT_EXTENSIONS, DocumentParser
from app.infrastructure.embedding_service import EmbeddingService
from app.infrastructure.es_client import ESClient
from app.infrastructure.vector_client import VectorClient
from app.schemas.structured import MetaRecord


class DocumentUpdater:
    def __init__(
        self,
        settings: Settings,
        es: ESClient,
        vectors: VectorClient,
        embed: EmbeddingService,
    ) -> None:
        self._settings = settings
        self._es = es
        self._vectors = vectors
        self._embed = embed
        self._parser = DocumentParser()
        self._chunker = Chunker(settings)
        self._meta = MetadataExtractor()
        self._cache = settings.cache_path

    def _cache_file(self, path: Path) -> Path:
        doc_id = Chunker.new_doc_id(path)
        return self._cache / f"{doc_id}.json"

    def _read_cache(self, path: Path) -> dict | None:
        for cache_path in (self._cache_file(path), self._cache / f"{path.name}.json"):
            if not cache_path.is_file():
                continue
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("invalid cache json {}", cache_path.name)
        return None

    def _write_cache(self, path: Path, payload: dict) -> None:
        cache_file = self._cache_file(path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        legacy = self._cache / f"{path.name}.json"
        if legacy != cache_file and legacy.is_file():
            legacy.unlink()

    async def _invalidate_text_doc(self, doc_id: str) -> None:
        logger.info("invalidating text doc_id={} in ES/Chroma…", doc_id)
        await asyncio.gather(
            self._es.mark_chunk_invalid(doc_id),
            self._vectors.mark_doc_chunks_invalid(doc_id),
        )

    async def _invalidate_meta_assets(self, asset_ids: list[str]) -> None:
        logger.info("invalidating {} structured asset(s) in ES/Chroma…", len(asset_ids))
        for asset_id in asset_ids:
            await self._es.mark_meta_invalid(asset_id)
            await self._vectors.mark_meta_invalid(asset_id)

    async def _invalidate_from_cache(self, cached: dict) -> None:
        kind = cached.get("kind")
        if kind == "text":
            doc_id = cached.get("doc_id")
            if doc_id:
                await self._invalidate_text_doc(doc_id)
        elif kind == "structured":
            asset_ids = cached.get("asset_ids") or []
            if asset_ids:
                await self._invalidate_meta_assets(asset_ids)

    async def _bulk_write_text_chunks(
        self,
        path_name: str,
        es_docs: list[dict],
        chroma_ids: list[str],
        embeddings: list[list[float]],
        chroma_docs: list[str],
        chroma_metas: list[dict],
    ) -> None:
        count = len(es_docs)
        t0 = time.perf_counter()
        logger.info("ingest text: {} — ES bulk ({} docs)…", path_name, count)
        await self._es.bulk_index_chunks(es_docs)
        logger.info("ingest text: {} — ES bulk done in {:.1f}s", path_name, time.perf_counter() - t0)
        t1 = time.perf_counter()
        logger.info("ingest text: {} — Chroma batch ({} docs)…", path_name, count)
        await self._vectors.upsert_chunks_batch(
            chroma_ids,
            embeddings,
            chroma_docs,
            chroma_metas,
        )
        logger.info("ingest text: {} — Chroma batch done in {:.1f}s", path_name, time.perf_counter() - t1)

    async def _bulk_write_meta_records(
        self,
        path_name: str,
        es_docs: list[dict],
        chroma_ids: list[str],
        embeddings: list[list[float]],
        chroma_docs: list[str],
        chroma_metas: list[dict],
    ) -> None:
        count = len(es_docs)
        t0 = time.perf_counter()
        logger.info("ingest structured: {} — ES bulk ({} docs)…", path_name, count)
        await self._es.bulk_index_meta(es_docs)
        logger.info("ingest structured: {} — ES bulk done in {:.1f}s", path_name, time.perf_counter() - t0)
        t1 = time.perf_counter()
        logger.info("ingest structured: {} — Chroma batch ({} docs)…", path_name, count)
        await self._vectors.upsert_meta_batch(
            chroma_ids,
            embeddings,
            chroma_docs,
            chroma_metas,
        )
        logger.info("ingest structured: {} — Chroma batch done in {:.1f}s", path_name, time.perf_counter() - t1)

    async def ingest_text_file(self, path: Path) -> None:
        logger.info("ingest text: {} — checking cache…", path.name)
        md5 = Chunker.file_md5(path)
        doc_id = Chunker.new_doc_id(path)
        source_path = str(path.resolve())
        cached = self._read_cache(path)

        if (
            cached
            and cached.get("md5") == md5
            and cached.get("kind") == "text"
            and not self._vectors.chunks_collection_was_reset()
        ):
            logger.info("skip unchanged text {}", path.name)
            return

        if cached and cached.get("doc_id"):
            logger.info("ingest text: {} — content changed, clearing old index…", path.name)
            await self._invalidate_text_doc(cached["doc_id"])

        logger.info("ingest text: {} — parsing document…", path.name)
        text = await self._parser.parse(path)
        chunks = self._chunker.chunk_text(text, source_file=source_path, doc_id=doc_id, md5=md5)
        texts = [c.content for c in chunks]
        if texts:
            logger.info(
                "ingest text: {} — {} chunk(s), embedding (model load may take a while, please wait)…",
                path.name,
                len(texts),
            )
            embeddings = await self._embed.embed(texts)
        else:
            embeddings = []
            logger.info("ingest text: {} — empty content, skipping embed", path.name)

        if chunks:
            es_docs = [
                {
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "status": chunk.status.value,
                    "source_file": chunk.source_file,
                    "asset_kind": chunk.asset_kind.value,
                }
                for chunk in chunks
            ]
            chroma_ids = [c.chunk_id for c in chunks]
            chroma_docs = [c.content for c in chunks]
            chroma_metas = [
                {
                    "doc_id": c.doc_id,
                    "source_file": c.source_file,
                    "status": c.status.value,
                }
                for c in chunks
            ]
            await self._bulk_write_text_chunks(
                path.name,
                es_docs,
                chroma_ids,
                embeddings,
                chroma_docs,
                chroma_metas,
            )
        self._write_cache(
            path,
            {
                "doc_id": doc_id,
                "md5": md5,
                "kind": "text",
                "source_path": source_path,
                "chunk_ids": [c.chunk_id for c in chunks],
            },
        )
        logger.info("ingested text {} chunks={}", path.name, len(chunks))
        gc.collect()

    async def ingest_structured_file(self, path: Path) -> None:
        logger.info("ingest structured: {} — checking cache…", path.name)
        md5 = Chunker.file_md5(path)
        doc_id = Chunker.new_doc_id(path)
        source_path = str(path.resolve())
        cached = self._read_cache(path)

        if cached and cached.get("md5") == md5 and cached.get("kind") == "structured":
            logger.info("skip unchanged structured {}", path.name)
            return

        old_asset_ids = list(cached.get("asset_ids") or []) if cached else []
        if old_asset_ids:
            logger.info("ingest structured: {} — content changed, clearing old index…", path.name)
            await self._invalidate_meta_assets(old_asset_ids)

        logger.info("ingest structured: {} — extracting metadata…", path.name)
        records = self._meta.extract(path)
        if not records:
            logger.warning("skip structured {} — no metadata extracted (empty or unsupported)", path.name)
            self._write_cache(
                path,
                {
                    "doc_id": doc_id,
                    "md5": md5,
                    "kind": "structured",
                    "source_path": source_path,
                    "asset_ids": [],
                    "count": 0,
                },
            )
            return
        logger.info(
            "ingest structured: {} — {} record(s), embedding + bulk index (please wait)…",
            path.name,
            len(records),
        )
        await self._index_meta_records(records, path.name)
        asset_ids = [r.asset_id for r in records]
        self._write_cache(
            path,
            {
                "doc_id": doc_id,
                "md5": md5,
                "kind": "structured",
                "source_path": source_path,
                "asset_ids": asset_ids,
                "count": len(records),
            },
        )
        logger.info("ingested structured meta {} records={}", path.name, len(records))

    async def _index_meta_records(self, records: list[MetaRecord], path_name: str = "") -> None:
        label = path_name or (records[0].file_name if records else "meta")
        search_texts = [r.search_text for r in records]
        embeddings = await self._embed.embed(search_texts)
        es_docs: list[dict] = []
        chroma_ids: list[str] = []
        chroma_docs: list[str] = []
        chroma_metas: list[dict] = []
        for record in records:
            doc = record.model_dump()
            doc["status"] = record.status.value
            doc["format"] = record.format.value
            doc["columns"] = record.columns
            es_docs.append(doc)
            chroma_ids.append(record.asset_id)
            chroma_docs.append(record.search_text)
            chroma_metas.append(
                {
                    "file_path": record.file_path,
                    "file_name": record.file_name,
                    "format": record.format.value,
                    "columns": "|".join(record.columns),
                    "status": record.status.value,
                }
            )
        await self._bulk_write_meta_records(
            label,
            es_docs,
            chroma_ids,
            embeddings,
            chroma_docs,
            chroma_metas,
        )

    async def _index_meta_record(self, record: MetaRecord) -> None:
        await self._index_meta_records([record], record.file_name)

    async def sync_path(self, path: Path) -> None:
        suffix = path.suffix.lower()
        if suffix in TEXT_EXTENSIONS:
            await self.ingest_text_file(path)
        elif suffix in STRUCTURED_EXTENSIONS:
            await self.ingest_structured_file(path)

    async def _invalidate_removed_under(self, directory: Path, seen_doc_ids: set[str]) -> None:
        if not self._cache.is_dir():
            return
        logger.info("sync: checking removed files under {}…", directory.name)
        root = directory.resolve()
        for cache_path in self._cache.glob("*.json"):
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            doc_id = cached.get("doc_id")
            if not doc_id or doc_id in seen_doc_ids:
                continue
            source = cached.get("source_path")
            if not source:
                continue
            try:
                src = Path(source).resolve()
            except OSError:
                continue
            if root not in src.parents and src != root:
                continue
            if src.is_file():
                continue
            logger.info("sync: source missing, invalidating {}…", src.name)
            await self._invalidate_from_cache(cached)
            logger.info("invalidated removed file {}", src.name)

    async def sync_directory(self, directory: Path) -> None:
        if not directory.exists():
            logger.info("sync: directory missing, skip {}", directory)
            return
        files = sorted(p for p in directory.rglob("*") if p.is_file())
        total = len(files)
        logger.info("sync: scanning {} — {} file(s) to process", directory, total)
        seen_doc_ids: set[str] = set()
        for idx, path in enumerate(files, start=1):
            logger.info("sync progress [{}/{}] {}", idx, total, path.name)
            seen_doc_ids.add(Chunker.new_doc_id(path))
            try:
                await self.sync_path(path)
            except Exception:
                logger.exception("sync failed for {} [{}/{}]", path.name, idx, total)
        logger.info("sync: finished {} [{}/{} files]", directory.name, total, total)
        await self._invalidate_removed_under(directory, seen_doc_ids)
        if directory == self._settings.raw_doc_path:
            self._vectors.ack_chunks_collection_reset()


async def sync_all(
    settings: Settings | None = None,
    *,
    es: ESClient | None = None,
    vectors: VectorClient | None = None,
    embed: EmbeddingService | None = None,
) -> None:
    """Full scan text + structured source directories.

    Reuse ``es`` / ``vectors`` / ``embed`` from AppContainer when provided to avoid
    opening a second Chroma PersistentClient (SQLite lock on Windows).
    """
    s = settings or get_settings()
    s.ensure_directories()
    own_es = es is None
    own_vectors = vectors is None
    if es is None:
        logger.info("sync_all: preparing ES indices and vector store…")
        es = ESClient(s)
        await es.ensure_indices()
    else:
        await es.ensure_indices()
    if vectors is None:
        vectors = VectorClient(s)
    if embed is None:
        embed = EmbeddingService(s)
    updater = DocumentUpdater(s, es, vectors, embed)
    try:
        logger.info("sync_all: text documents → {}", s.raw_doc_path)
        await updater.sync_directory(s.raw_doc_path)
        logger.info("sync_all: structured metadata → {}", s.raw_structured_path)
        await updater.sync_directory(s.raw_structured_path)
        logger.info("sync_all: completed")
    finally:
        if own_es:
            await es.close()
        if own_vectors:
            vectors.close()
