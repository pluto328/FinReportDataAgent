"""Loguru logger with rotation and truncation."""

from __future__ import annotations

import sys
from typing import Any

from loguru import logger

MAX_LOG_TEXT = 500


def _truncate(record: dict[str, Any]) -> None:
    message = record["message"]
    if isinstance(message, str) and len(message) > MAX_LOG_TEXT:
        record["message"] = f"{message[:MAX_LOG_TEXT]}...(truncated,{len(message)} chars)"


def setup_logger(*, log_dir: str, level: str = "INFO") -> None:
    """Configure loguru sinks; idempotent."""
    logger.remove()
    logger.configure(patcher=_truncate)
    logger.add(sys.stderr, level=level, enqueue=True)
    logger.add(
        f"{log_dir}/app_{{time:YYYY-MM-DD}}.log",
        rotation="00:00",
        retention="14 days",
        level=level,
        enqueue=True,
        encoding="utf-8",
    )


__all__ = ["logger", "setup_logger"]
