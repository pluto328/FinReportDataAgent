"""跨层公共工具。

logger.py 规约：
- loguru 分级滚动日志，生产默认 INFO
- 正文/Prompt/检索结果等长文本截断或只记长度/ID，禁止全量刷屏
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["logger"]

if TYPE_CHECKING:
    from app.common.logger import logger as logger


def __getattr__(name: str) -> Any:
    if name == "logger":
        from app.common.logger import logger

        return logger
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
