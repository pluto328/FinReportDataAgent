"""文档解析、切块、增量同步。

子模块规约：
- parser.py：pypdf/python-docx 提取正文；仅提取正文，不参与 Tool 注册
- chunker.py：CHUNK_SIZE/CHUNK_OVERLAP 滑窗 → DocChunk 列表
- updater.py：MD5 比对；变更/删除 → 旧 chunk 标记 invalid；sync_all() 全量兜底扫描

禁止物理删除 parsed_cache 或向量文件。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["Chunker", "DocumentParser", "DocumentUpdater", "sync_all"]

_EXPORTS: dict[str, tuple[str, str]] = {
    "DocumentParser": ("app.core.ingestion.parser", "DocumentParser"),
    "Chunker": ("app.core.ingestion.chunker", "Chunker"),
    "DocumentUpdater": ("app.core.ingestion.updater", "DocumentUpdater"),
    "sync_all": ("app.core.ingestion.updater", "sync_all"),
}

if TYPE_CHECKING:
    from app.core.ingestion.chunker import Chunker
    from app.core.ingestion.parser import DocumentParser
    from app.core.ingestion.updater import DocumentUpdater, sync_all


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    module_path, attr = _EXPORTS[name]
    module = __import__(module_path, fromlist=[attr])
    return getattr(module, attr)
