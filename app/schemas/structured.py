"""Structured metadata, agent flags, session cache, report artifacts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.document import ChunkStatus


class StructuredFormat(StrEnum):
    CSV = "csv"
    TSV = "tsv"
    XLSX = "xlsx"
    XLSB = "xlsb"
    PARQUET = "parquet"
    FEATHER = "feather"
    JSONL = "jsonl"


class MetaRecord(BaseModel):
    """Metadata-only index record; no row data."""

    asset_id: str
    file_path: str
    file_name: str
    format: StructuredFormat
    table_name: str = ""
    sheet_name: str = ""
    columns: list[str] = Field(default_factory=list)
    time_columns: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    search_text: str = ""
    status: ChunkStatus = ChunkStatus.ONLINE
    version: int = 1
    md5: str = ""


class ScoredMetaRecord(BaseModel):
    record: MetaRecord
    score: float
    retriever: str = ""


DataProcessMode = Literal["none", "tool", "sql"]


class NodeEnableFlags(BaseModel):
    enable_knowledge_retrieve: bool = True
    enable_data_retrieve: bool = False
    enable_process: bool = False
    enable_chart: bool = False
    enable_report: bool = False


class ProcessedDataRef(BaseModel):
    """On-disk processed artifact; state keeps path + preview only."""

    path: str
    preview: str = ""
    mode: DataProcessMode = "tool"
    row_count: int = 0
    byte_size: int = 0
    source_file: str = ""


class ChartType(StrEnum):
    TABLE = "table"
    LINE = "line"
    BAR = "bar"


class ChartSpec(BaseModel):
    need_chart: bool = False
    chart_type: ChartType = ChartType.TABLE
    x_axis: str = ""
    y_axis: str = ""
    title: str = ""


class ChartArtifact(BaseModel):
    path: str
    description: str = ""
    title: str = ""
    chart_type: str = "table"


class PendingToolCall(BaseModel):
    """Unified pending tool invocation for ReAct loops across plan/data/report phases."""

    phase: Literal["plan", "data", "report"]
    tool_name: str
    params: dict[str, Any] = Field(default_factory=dict)
    file_path: str = ""


class PlanStepResult(BaseModel):
    step: int
    method: str
    params: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class DataToolStepResult(BaseModel):
    step: int
    tool_name: str
    params: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    error: str = ""
    artifact_ref: ProcessedDataRef | None = None


class FilePreviewEntry(BaseModel):
    """Per-file preview cache for agent prompts (up to 20 rows on disk in state)."""

    path: str
    description: str = ""
    preview_rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0


FILE_PREVIEW_STORE_ROWS = 20
DATA_PROCESSOR_PREVIEW_DISPLAY_ROWS = 3


class ReportArtifact(BaseModel):
    session_id: str
    markdown_path: str = ""
    html_path: str = ""
    table_paths: list[str] = Field(default_factory=list)
    image_paths: list[str] = Field(default_factory=list)
    reference_docs: list[str] = Field(default_factory=list)
    nodes_traversed: list[str] = Field(default_factory=list)


class ReportStepResult(BaseModel):
    step: int
    tool_name: str
    params: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


AgentStatus = Literal["ok", "not_found", "partial", "rejected"]
