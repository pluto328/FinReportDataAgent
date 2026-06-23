"""LLM connection warmup: startup ping + in-pipeline ping (no scheduled ping)."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from app.common.logger import logger

if TYPE_CHECKING:
    from app.config.settings import Settings
    from app.core.agent.state import AgentRuntime
    from app.infrastructure.llm_client import LLMClient

_warmed_at: dict[int, float] = {}


def _warmup_enabled(settings: Settings) -> bool:
    if not settings.llm_warmup_enabled:
        return False
    if not (settings.llm_base_url or "").strip():
        return False
    if not (settings.llm_api_key or "").strip():
        return False
    return True


async def _ping_client(client: LLMClient, settings: Settings, *, role: str) -> bool:
    if not _warmup_enabled(settings):
        return False
    try:
        await asyncio.wait_for(client.ping(), timeout=settings.llm_warmup_timeout_sec)
        _warmed_at[id(client)] = time.monotonic()
        logger.info("LLM warmup ok role={} model={}", role, client.model)
        return True
    except Exception as exc:
        logger.warning("LLM warmup failed role={} model={}: {}", role, client.model, exc)
        return False


def _distinct_role_clients(runtime: AgentRuntime) -> list[tuple[str, LLMClient]]:
    seen: set[int] = set()
    out: list[tuple[str, LLMClient]] = []
    for role, client in (
        ("planner", runtime.llm_for_planner()),
        ("data", runtime.llm_for_data()),
        ("reporter", runtime.llm_for_reporter()),
    ):
        cid = id(client)
        if cid in seen:
            continue
        seen.add(cid)
        out.append((role, client))
    return out


async def warmup_startup(runtime: AgentRuntime, settings: Settings) -> None:
    """Ping each role client once at application startup."""
    if not _warmup_enabled(settings):
        logger.debug("LLM startup warmup skipped (disabled or missing API config)")
        return
    clients = _distinct_role_clients(runtime)
    logger.info("LLM startup warmup for {} client(s)", len(clients))
    await asyncio.gather(
        *(_ping_client(client, settings, role=role) for role, client in clients),
        return_exceptions=True,
    )


def schedule_pipeline_data_warmup(runtime: AgentRuntime, settings: Settings) -> None:
    """Fire-and-forget data LLM ping while planner runs."""
    if not _warmup_enabled(settings) or not settings.llm_warmup_pipeline:
        return
    client = runtime.llm_for_data()

    async def _run() -> None:
        await _ping_client(client, settings, role="data")

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_run())
    except RuntimeError:
        pass


async def warmup_pipeline_reporter(runtime: AgentRuntime, settings: Settings) -> None:
    """Ping reporter LLM immediately before the reporter decision call."""
    if not _warmup_enabled(settings) or not settings.llm_warmup_pipeline:
        return
    await _ping_client(runtime.llm_for_reporter(), settings, role="reporter")
