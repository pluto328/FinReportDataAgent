"""Watch raw_docs and raw_structured for changes; debounce and trigger incremental sync."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.common.logger import logger, setup_logger
from app.config.settings import get_settings
from app.core.ingestion.meta_extractor import STRUCTURED_EXTENSIONS
from app.core.ingestion.parser import TEXT_EXTENSIONS
from app.core.ingestion.updater import DocumentUpdater, sync_all
from app.infrastructure.embedding_service import EmbeddingService
from app.infrastructure.es_client import ESClient
from app.infrastructure.vector_client import VectorClient

SUPPORTED = TEXT_EXTENSIONS | STRUCTURED_EXTENSIONS
DEBOUNCE_SEC = 2.0


class IngestHandler(FileSystemEventHandler):
    def __init__(self, loop: asyncio.AbstractEventLoop, updater: DocumentUpdater) -> None:
        self._loop = loop
        self._updater = updater
        self._pending: dict[str, float] = {}

    def on_created(self, event: FileSystemEvent) -> None:
        self._schedule(event)

    def on_modified(self, event: FileSystemEvent) -> None:
        self._schedule(event)

    def _schedule(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() not in SUPPORTED:
            return
        self._pending[str(path)] = time.monotonic()
        self._loop.call_later(DEBOUNCE_SEC, self._flush, str(path))

    def _flush(self, key: str) -> None:
        ts = self._pending.get(key)
        if ts is None:
            return
        if time.monotonic() - ts < DEBOUNCE_SEC - 0.05:
            return
        self._pending.pop(key, None)
        path = Path(key)
        if not path.exists() or not path.is_file():
            return
        logger.info("monitor sync {}", path.name)
        asyncio.run_coroutine_threadsafe(self._updater.sync_path(path), self._loop)


async def _build_updater() -> tuple[DocumentUpdater, ESClient, VectorClient, EmbeddingService]:
    settings = get_settings()
    settings.ensure_directories()
    es = ESClient(settings)
    await es.ensure_indices()
    vectors = VectorClient(settings)
    embed = EmbeddingService(settings)
    updater = DocumentUpdater(settings, es, vectors, embed)
    return updater, es, vectors, embed


def main() -> None:
    settings = get_settings()
    settings.ensure_directories()
    setup_logger(log_dir=str(settings.log_dir), level=settings.log_level)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    updater, es, vectors, embed = loop.run_until_complete(_build_updater())
    loop.run_until_complete(
        sync_all(
            settings,
            es=es,
            vectors=vectors,
            embed=embed,
        )
    )

    handler = IngestHandler(loop, updater)
    observer = Observer()
    for directory in (settings.raw_doc_path, settings.raw_structured_path):
        directory.mkdir(parents=True, exist_ok=True)
        observer.schedule(handler, str(directory), recursive=True)
        logger.info("watching {}", directory)

    observer.start()
    logger.info("monitor started (debounce={}s)", DEBOUNCE_SEC)
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        logger.info("monitor stopping")
    finally:
        observer.stop()
        observer.join()
        loop.run_until_complete(es.close())
        vectors.close()
        loop.close()


if __name__ == "__main__":
    main()
