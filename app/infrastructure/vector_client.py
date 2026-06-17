"""Chroma vector store with dual collections (HTTP server or local persistent)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING, TypeVar

from app.common.logger import logger
from app.config.settings import Settings
from app.infrastructure.chroma_bootstrap import ensure_chroma_without_onnx

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection

CHROMA_UPSERT_BATCH_SIZE = 16
CHROMA_UPSERT_TIMEOUT_SEC = 120
_CHROMA_EXECUTOR: concurrent.futures.ThreadPoolExecutor | None = None
_CHROMA_EXECUTOR_INIT = threading.Lock()
T = TypeVar("T")


def _get_chroma_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _CHROMA_EXECUTOR
    with _CHROMA_EXECUTOR_INIT:
        if _CHROMA_EXECUTOR is None:
            _CHROMA_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="chroma-io",
            )
        return _CHROMA_EXECUTOR


def _explicit_embedding_function():
    ensure_chroma_without_onnx()
    from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

    class ExplicitEmbeddingFunction(EmbeddingFunction[Documents]):
        def __call__(self, input: Documents) -> Embeddings:
            raise RuntimeError(
                "Chroma default embeddings are disabled; supply embeddings via EmbeddingService."
            )

    return ExplicitEmbeddingFunction()


def _chroma_settings() -> "ChromaSettings":
    from chromadb.config import Settings as ChromaSettings

    return ChromaSettings(
        anonymized_telemetry=False,
        chroma_product_telemetry_impl=(
            "app.infrastructure.chroma_telemetry.NoOpProductTelemetry"
        ),
    )


def _sanitize_metadata(meta: dict[str, Any]) -> dict[str, str | int | float | bool]:
    clean: dict[str, str | int | float | bool] = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def _create_client(settings: Settings):
    ensure_chroma_without_onnx()
    import chromadb

    embedding_function = _explicit_embedding_function()
    chroma_settings = _chroma_settings()

    if settings.chroma_use_http:
        client = chromadb.HttpClient(
            host=settings.chroma_http_host,
            port=settings.chroma_http_port,
            settings=chroma_settings,
        )
        client.heartbeat()
        logger.info(
            "Chroma HTTP client → {}:{}",
            settings.chroma_http_host,
            settings.chroma_http_port,
        )
    else:
        client = chromadb.PersistentClient(
            path=str(settings.vector_persist_path),
            settings=chroma_settings,
        )
        logger.info("Chroma PersistentClient → {}", settings.vector_persist_path)

    return client, embedding_function


@dataclass
class _ChromaHandles:
    client: Any
    chunk: Any
    meta: Any
    embedding_function: Any


def _bootstrap_chroma(settings: Settings) -> _ChromaHandles:
    client, embedding_function = _create_client(settings)
    chunk = client.get_or_create_collection(
        settings.chroma_collection,
        embedding_function=embedding_function,
    )
    meta = client.get_or_create_collection(
        settings.chroma_meta_collection,
        embedding_function=embedding_function,
    )
    return _ChromaHandles(
        client=client,
        chunk=chunk,
        meta=meta,
        embedding_function=embedding_function,
    )


class VectorClient:
    """Chroma client for chunk and meta embeddings."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._chunk_collection_name = settings.chroma_collection
        self._meta_collection_name = settings.chroma_meta_collection
        self._chunks_collection_was_reset = False
        self._process_lock = None

        if not settings.chroma_use_http:
            from app.infrastructure.chroma_lock import ChromaProcessLock

            self._process_lock = ChromaProcessLock(Path(settings.vector_persist_path))
            self._process_lock.acquire()

        handles = _get_chroma_executor().submit(_bootstrap_chroma, settings).result(timeout=120)
        self._client = handles.client
        self._chunk = handles.chunk
        self._meta = handles.meta
        self._embedding_function = handles.embedding_function

    async def _chroma_call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _get_chroma_executor(),
            lambda: fn(*args, **kwargs),
        )

    def chunks_collection_was_reset(self) -> bool:
        return self._chunks_collection_was_reset

    def ack_chunks_collection_reset(self) -> None:
        self._chunks_collection_was_reset = False

    def _reset_chunk_collection_sync(self) -> None:
        logger.warning("recreating Chroma chunk collection {}", self._chunk_collection_name)
        try:
            self._client.delete_collection(self._chunk_collection_name)
        except Exception:
            pass
        self._chunk = self._client.get_or_create_collection(
            self._chunk_collection_name,
            embedding_function=self._embedding_function,
        )
        self._chunks_collection_was_reset = True

    def _reset_meta_collection_sync(self) -> None:
        logger.warning("recreating Chroma meta collection {}", self._meta_collection_name)
        try:
            self._client.delete_collection(self._meta_collection_name)
        except Exception:
            pass
        self._meta = self._client.get_or_create_collection(
            self._meta_collection_name,
            embedding_function=self._embedding_function,
        )

    async def _reset_chunk_collection(self) -> None:
        await self._chroma_call(self._reset_chunk_collection_sync)

    async def _reset_meta_collection(self) -> None:
        await self._chroma_call(self._reset_meta_collection_sync)

    def close(self) -> None:
        def _release() -> None:
            self._chunk = None
            self._meta = None
            self._client = None

        try:
            _get_chroma_executor().submit(_release).result(timeout=5)
        except Exception:
            pass
        if self._process_lock is not None:
            self._process_lock.release()

    async def upsert_chunk(
        self,
        chunk_id: str,
        embedding: list[float],
        document: str,
        metadata: dict[str, Any],
    ) -> None:
        await self.upsert_chunks_batch(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[metadata],
        )

    async def upsert_meta(
        self,
        asset_id: str,
        embedding: list[float],
        document: str,
        metadata: dict[str, Any],
    ) -> None:
        await self.upsert_meta_batch(
            ids=[asset_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[metadata],
        )

    async def upsert_chunks_batch(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        await self._upsert_batched(
            "chunk",
            lambda: self._chunk,
            ids,
            embeddings,
            documents,
            metadatas,
        )

    async def upsert_meta_batch(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        await self._upsert_batched(
            "meta",
            lambda: self._meta,
            ids,
            embeddings,
            documents,
            metadatas,
        )

    async def _upsert_batched(
        self,
        label: str,
        get_collection: Callable[[], "Collection"],
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not ids:
            return
        clean_metas = [_sanitize_metadata(m) for m in metadatas]
        batch_size = CHROMA_UPSERT_BATCH_SIZE
        total = len(ids)
        batch_count = (total + batch_size - 1) // batch_size
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_idx = start // batch_size + 1
            await self._upsert_one_batch(
                label,
                get_collection,
                ids[start:end],
                embeddings[start:end],
                documents[start:end],
                clean_metas[start:end],
                batch_idx=batch_idx,
                batch_count=batch_count,
                allow_reset_retry=True,
            )

    def _upsert_batch_sync(
        self,
        collection: "Collection",
        label: str,
        ids_batch: list[str],
        embs_batch: list[list[float]],
        docs_batch: list[str],
        metas_batch: list[dict[str, str | int | float | bool]],
        batch_idx: int,
        batch_count: int,
    ) -> None:
        t0 = time.perf_counter()
        logger.info(
            "Chroma {} upsert sub-batch {}/{} ({} docs)…",
            label,
            batch_idx,
            batch_count,
            len(ids_batch),
        )
        collection.upsert(
            ids=ids_batch,
            embeddings=embs_batch,
            documents=docs_batch,
            metadatas=metas_batch,
        )
        logger.info(
            "Chroma {} upsert sub-batch {}/{} done in {:.1f}s",
            label,
            batch_idx,
            batch_count,
            time.perf_counter() - t0,
        )

    async def _upsert_one_batch(
        self,
        label: str,
        get_collection: Callable[[], "Collection"],
        ids_batch: list[str],
        embs_batch: list[list[float]],
        docs_batch: list[str],
        metas_batch: list[dict[str, str | int | float | bool]],
        *,
        batch_idx: int,
        batch_count: int,
        allow_reset_retry: bool,
    ) -> None:
        collection = get_collection()
        try:
            await asyncio.wait_for(
                self._chroma_call(
                    self._upsert_batch_sync,
                    collection,
                    label,
                    ids_batch,
                    embs_batch,
                    docs_batch,
                    metas_batch,
                    batch_idx,
                    batch_count,
                ),
                timeout=CHROMA_UPSERT_TIMEOUT_SEC,
            )
        except (asyncio.TimeoutError, Exception) as exc:
            logger.error(
                "Chroma {} upsert sub-batch {}/{} failed: {}",
                label,
                batch_idx,
                batch_count,
                exc,
            )
            if not allow_reset_retry:
                raise RuntimeError(
                    f"Chroma {label} upsert failed (batch {batch_idx}/{batch_count})"
                ) from exc
            if label == "chunk":
                await self._reset_chunk_collection()
            else:
                await self._reset_meta_collection()
            await self._upsert_one_batch(
                label,
                get_collection,
                ids_batch,
                embs_batch,
                docs_batch,
                metas_batch,
                batch_idx=batch_idx,
                batch_count=batch_count,
                allow_reset_retry=False,
            )

    async def query_chunks(
        self,
        embedding: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        result = await self._chroma_call(
            self._chunk.query,
            query_embeddings=[embedding],
            n_results=top_k,
            where={"status": "online"},
        )
        return self._parse_query_result(result)

    async def query_meta(
        self,
        embedding: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        result = await self._chroma_call(
            self._meta.query,
            query_embeddings=[embedding],
            n_results=top_k,
            where={"status": "online"},
        )
        return self._parse_query_result(result)

    async def mark_chunk_invalid(self, chunk_id: str) -> None:
        await self._chroma_call(
            self._chunk.update,
            ids=[chunk_id],
            metadatas=[{"status": "invalid"}],
        )

    async def mark_doc_chunks_invalid(self, doc_id: str) -> None:
        def _run() -> None:
            result = self._chunk.get(where={"doc_id": doc_id}, include=["metadatas"])
            ids = result.get("ids") or []
            if not ids:
                return
            metas = result.get("metadatas") or []
            updated = []
            for meta in metas:
                merged = dict(meta or {})
                merged["status"] = "invalid"
                updated.append(merged)
            self._chunk.update(ids=ids, metadatas=updated)

        await self._chroma_call(_run)

    async def mark_meta_invalid(self, asset_id: str) -> None:
        def _run() -> None:
            try:
                result = self._meta.get(ids=[asset_id], include=["metadatas"])
            except Exception:
                return
            ids = result.get("ids") or []
            if not ids:
                return
            meta = dict((result.get("metadatas") or [{}])[0] or {})
            meta["status"] = "invalid"
            self._meta.update(ids=ids, metadatas=[meta])

        await self._chroma_call(_run)

    @staticmethod
    def _parse_query_result(result: dict[str, Any]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for i, doc_id in enumerate(ids):
            score = 1.0 / (1.0 + dists[i]) if dists else 0.0
            items.append(
                {
                    "id": doc_id,
                    "document": docs[i] if docs else "",
                    "metadata": metas[i] if metas else {},
                    "_score": score,
                }
            )
        return items
