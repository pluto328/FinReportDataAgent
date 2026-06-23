"""Pydantic 数据契约：文本 chunk、结构化元数据、API、评测。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "AgentStatus",
    "AssetKind",
    "ChartSpec",
    "ChartType",
    "ChatMessage",
    "ChunkStatus",
    "DocChunk",
    "DocumentCache",
    "EvalMetrics",
    "MetaRecord",
    "NodeEnableFlags",
    "PendingToolCall",
    "PlanStepResult",
    "DataToolStepResult",
    "ReportArtifact",
    "ReportRequest",
    "ReportResponse",
    "ScoredChunk",
    "ScoredMetaRecord",
    "SearchRequest",
    "SearchResponse",
    "StructuredFormat",
    "UploadResponse",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "AssetKind": ("app.schemas.document", "AssetKind"),
    "ChunkStatus": ("app.schemas.document", "ChunkStatus"),
    "DocChunk": ("app.schemas.document", "DocChunk"),
    "DocumentCache": ("app.schemas.document", "DocumentCache"),
    "ScoredChunk": ("app.schemas.document", "ScoredChunk"),
    "StructuredFormat": ("app.schemas.structured", "StructuredFormat"),
    "MetaRecord": ("app.schemas.structured", "MetaRecord"),
    "ScoredMetaRecord": ("app.schemas.structured", "ScoredMetaRecord"),
    "NodeEnableFlags": ("app.schemas.structured", "NodeEnableFlags"),
    "PendingToolCall": ("app.schemas.structured", "PendingToolCall"),
    "ChartType": ("app.schemas.structured", "ChartType"),
    "ChartSpec": ("app.schemas.structured", "ChartSpec"),
    "PlanStepResult": ("app.schemas.structured", "PlanStepResult"),
    "DataToolStepResult": ("app.schemas.structured", "DataToolStepResult"),
    "ReportArtifact": ("app.schemas.structured", "ReportArtifact"),
    "AgentStatus": ("app.schemas.structured", "AgentStatus"),
    "ChatMessage": ("app.schemas.query", "ChatMessage"),
    "SearchRequest": ("app.schemas.query", "SearchRequest"),
    "SearchResponse": ("app.schemas.query", "SearchResponse"),
    "ReportRequest": ("app.schemas.query", "ReportRequest"),
    "ReportResponse": ("app.schemas.query", "ReportResponse"),
    "UploadResponse": ("app.schemas.query", "UploadResponse"),
    "EvalMetrics": ("app.schemas.metrics", "EvalMetrics"),
}

if TYPE_CHECKING:
    from app.schemas.document import AssetKind, ChunkStatus, DocChunk, DocumentCache, ScoredChunk
    from app.schemas.metrics import EvalMetrics
    from app.schemas.query import (
        ChatMessage,
        ReportRequest,
        ReportResponse,
        SearchRequest,
        SearchResponse,
        UploadResponse,
    )
    from app.schemas.structured import (
        AgentStatus,
        ChartSpec,
        ChartType,
        DataToolStepResult,
        MetaRecord,
        NodeEnableFlags,
        PendingToolCall,
        PlanStepResult,
        ReportArtifact,
        ScoredMetaRecord,
        StructuredFormat,
    )


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    module_path, attr = _EXPORTS[name]
    module = __import__(module_path, fromlist=[attr])
    return getattr(module, attr)
