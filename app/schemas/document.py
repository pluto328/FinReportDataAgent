"""Text document and chunk models."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class AssetKind(StrEnum):
    TEXT = "text"
    STRUCTURED = "structured"


class ChunkStatus(StrEnum):
    ONLINE = "online"
    INVALID = "invalid"


class DocChunk(BaseModel):
    """Standard text chunk stored in ES / Chroma."""

    doc_id: str
    chunk_id: str
    content: str
    asset_kind: AssetKind = AssetKind.TEXT
    status: ChunkStatus = ChunkStatus.ONLINE
    version: int = 1
    md5: str = ""
    source_file: str = ""
    chunk_index: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentCache(BaseModel):
    """Parsed document cache entry."""

    doc_id: str
    file_path: str
    md5: str
    asset_kind: AssetKind
    status: ChunkStatus = ChunkStatus.ONLINE
    version: int = 1
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ScoredChunk(BaseModel):
    """Retrieval result for text chunks."""

    chunk: DocChunk
    score: float
    retriever: str = ""
