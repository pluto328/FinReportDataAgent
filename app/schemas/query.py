"""API request / response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.schemas.document import ScoredChunk
from app.schemas.structured import AgentStatus, ReportArtifact, ScoredMetaRecord


class ChatMessage(BaseModel):
    role: str
    content: str


class SearchRequest(BaseModel):
    query: str
    session_id: str = ""
    chat_history: list[ChatMessage] = Field(default_factory=list)
    report_mode: bool = False
    new_session: bool = False
    sources: list[str] = Field(default_factory=lambda: ["text", "structured"])


class SearchResponse(BaseModel):
    answer: str = ""
    message: str = ""
    session_id: str = ""
    status: AgentStatus = "ok"
    report_url: str = ""
    knowledge_chunks: list[ScoredChunk] = Field(default_factory=list)
    meta_hits: list[ScoredMetaRecord] = Field(default_factory=list)
    chart_artifacts: list[str] = Field(default_factory=list)


class ReportRequest(BaseModel):
    query: str
    session_id: str = ""
    chat_history: list[ChatMessage] = Field(default_factory=list)


class ReportResponse(BaseModel):
    message: str = "详见报告"
    session_id: str = ""
    report_url: str = ""
    download_url: str = ""
    artifact: ReportArtifact | None = None
    status: AgentStatus = "ok"


class UploadResponse(BaseModel):
    doc_id: str
    file_name: str
    asset_kind: str
    message: str = "uploaded"
