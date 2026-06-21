"""FastAPI application entry."""

from __future__ import annotations

import asyncio
import faulthandler
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

from app.config.settings import get_settings
from app.infrastructure.hf_hub_config import configure_hf_hub

configure_hf_hub(get_settings())
faulthandler.enable(file=sys.stderr, all_threads=True)

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import mount_routes
from app.common.logger import logger, setup_logger
from app.core.ingestion.updater import sync_all
from app.dependencies import get_container, init_container


async def _run_startup_sync(settings, container) -> None:
    try:
        await sync_all(
            settings,
            es=container.es,
            vectors=container.vectors,
            embed=container.embed,
        )
        logger.info("sync_all completed")
    except Exception:
        logger.exception("sync_all failed")


async def _run_embed_warmup(container) -> None:
    try:
        await container.embed.warmup()
    except Exception:
        logger.exception("embed warmup failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_hf_hub(settings)
    settings.ensure_directories()
    setup_logger(log_dir=str(settings.log_dir), level=settings.log_level)
    container = init_container(settings)
    sync_task: asyncio.Task | None = None
    warmup_task = asyncio.create_task(_run_embed_warmup(container))
    logger.info("startup embed warmup (background)")

    if settings.sync_on_startup:
        if settings.sync_in_background:
            logger.info("startup sync_all (background)")
            sync_task = asyncio.create_task(_run_startup_sync(settings, container))
            logger.info(
                "server ready on {}:{} — ingestion continues in background",
                settings.app_host,
                settings.app_port,
            )
        else:
            logger.info("startup sync_all (blocking)")
            await _run_startup_sync(settings, container)
    else:
        logger.info("startup sync skipped (SYNC_ON_STARTUP=false)")

    app.state.sync_task = sync_task
    app.state.warmup_task = warmup_task
    yield

    if warmup_task is not None and not warmup_task.done():
        logger.info("shutdown: waiting for embed warmup (max 120s)…")
        try:
            await asyncio.wait_for(warmup_task, timeout=120)
        except asyncio.TimeoutError:
            logger.warning("embed warmup still running; cancelling")
            warmup_task.cancel()
            try:
                await warmup_task
            except asyncio.CancelledError:
                pass

    if sync_task is not None and not sync_task.done():
        logger.info("shutdown: waiting for background sync (max 60s)…")
        try:
            await asyncio.wait_for(sync_task, timeout=60)
        except asyncio.TimeoutError:
            logger.warning("background sync still running; cancelling")
            sync_task.cancel()
            try:
                await sync_task
            except asyncio.CancelledError:
                pass

    container = get_container()
    await container.shutdown()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Knowledge RAG System Agent", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    mount_routes(app.router)
    return app


app = create_app()
