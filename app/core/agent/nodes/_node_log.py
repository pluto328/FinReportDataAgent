"""Structured INFO logging for LangGraph agent nodes."""

from __future__ import annotations

import json
from typing import Any

from app.common.logger import logger
from app.config.settings import Settings

_LEVEL_RANK = {"OFF": 0, "ERROR": 1, "INFO": 2, "DEBUG": 3}
_TRUNC = 320


def _fmt_ctx(ctx: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, val in ctx.items():
        if val is None:
            continue
        if hasattr(val, "model_dump"):
            text = json.dumps(val.model_dump(), ensure_ascii=False)
        elif isinstance(val, (dict, list)):
            text = json.dumps(val, ensure_ascii=False, default=str)
        else:
            text = str(val)
        if len(text) > _TRUNC:
            text = f"{text[:_TRUNC]}…"
        parts.append(f"{key}={text}")
    return " ".join(parts)


class NodeLogger:
    """Per-node logger gated by Settings.enable_agent_node_log / agent_node_log_level."""

    def __init__(self, settings: Settings, node: str) -> None:
        self.settings = settings
        self.node = node

    def _rank(self) -> int:
        if not self.settings.enable_agent_node_log:
            return 0
        level = (self.settings.agent_node_log_level or "INFO").upper()
        return _LEVEL_RANK.get(level, _LEVEL_RANK["INFO"])

    def _emit(self, min_rank: int, level: str, message: str, **ctx: Any) -> None:
        if self._rank() < min_rank:
            return
        suffix = _fmt_ctx(ctx)
        text = f"[agent][{self.node}] {message}"
        if suffix:
            text = f"{text} {suffix}"
        getattr(logger, level)(text)

    def start(self, **ctx: Any) -> None:
        self._emit(_LEVEL_RANK["INFO"], "info", "START", **ctx)

    def info(self, message: str, **ctx: Any) -> None:
        self._emit(_LEVEL_RANK["INFO"], "info", message, **ctx)

    def debug(self, message: str, **ctx: Any) -> None:
        self._emit(_LEVEL_RANK["DEBUG"], "debug", message, **ctx)

    def end(self, **ctx: Any) -> None:
        self._emit(_LEVEL_RANK["INFO"], "info", "END", **ctx)

    def fail(self, message: str, **ctx: Any) -> None:
        self._emit(_LEVEL_RANK["ERROR"], "error", f"END {message}", **ctx)


def node_logger(settings: Settings, node: str) -> NodeLogger:
    return NodeLogger(settings, node)
