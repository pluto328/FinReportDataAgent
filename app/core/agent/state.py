"""LangGraph agent state."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any

from typing_extensions import TypedDict

from app.core.retrieval.ensemble import EnsembleRetriever
from app.core.retrieval.meta_ensemble import MetaEnsembleRetriever
from app.core.tools.registry import ToolRegistry
from app.infrastructure.llm_client import LLMClient
from app.schemas.document import ScoredChunk
from app.schemas.query import ChatMessage
from app.schemas.structured import (
    AgentStatus,
    ChartArtifact,
    DataToolStepResult,
    FilePreviewEntry,
    NodeEnableFlags,
    PendingToolCall,
    PlanStepResult,
    ProcessedDataRef,
    ReportArtifact,
    ReportStepResult,
    ScoredMetaRecord,
    SessionFileCache,
)
from app.config.settings import Settings


def _merge_file_previews(
    left: dict[str, FilePreviewEntry] | None,
    right: dict[str, FilePreviewEntry] | None,
) -> dict[str, FilePreviewEntry]:
    merged = dict(left or {})
    if right:
        merged.update(right)
    return merged


def _merge_max_int(left: int | None, right: int | None) -> int:
    return max(left or 0, right or 0)


def _chunk_dedup_key(chunk: ScoredChunk) -> str:
    cid = str(chunk.chunk.chunk_id or "").strip()
    if cid:
        return cid
    src = str(chunk.chunk.source_file or chunk.chunk.doc_id or "").strip()
    idx = chunk.chunk.chunk_index
    content_sig = hash(chunk.chunk.content[:240])
    return f"{src}:{idx}:{content_sig}"


def _merge_knowledge_chunks(
    left: list[ScoredChunk] | None,
    right: list[ScoredChunk] | None,
) -> list[ScoredChunk]:
    by_key: dict[str, ScoredChunk] = {}
    for item in (left or []) + (right or []):
        key = _chunk_dedup_key(item)
        prev = by_key.get(key)
        if prev is None or item.score > prev.score:
            by_key[key] = item
    return list(by_key.values())


def _meta_dedup_key(record: ScoredMetaRecord) -> str:
    path = str(record.record.file_path or "").strip()
    if path:
        return path.lower()
    asset = str(record.record.asset_id or "").strip()
    if asset:
        return asset
    name = str(record.record.file_name or "").strip()
    return name.lower() if name else str(id(record))


def _merge_meta_hits(
    left: list[ScoredMetaRecord] | None,
    right: list[ScoredMetaRecord] | None,
) -> list[ScoredMetaRecord]:
    by_key: dict[str, ScoredMetaRecord] = {}
    for item in (left or []) + (right or []):
        key = _meta_dedup_key(item)
        prev = by_key.get(key)
        if prev is None or item.score > prev.score:
            by_key[key] = item
    return list(by_key.values())


def _merge_unique_paths(left: list[str] | None, right: list[str] | None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in (left or []) + (right or []):
        path = str(raw or "").strip()
        if not path:
            continue
        key = path.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def dedupe_scored_chunks(chunks: list[ScoredChunk]) -> list[ScoredChunk]:
    return _merge_knowledge_chunks([], chunks)


def dedupe_scored_meta(meta: list[ScoredMetaRecord]) -> list[ScoredMetaRecord]:
    return _merge_meta_hits([], meta)


class AgentState(TypedDict, total=False):
    user_query: str
    user_require: str
    chat_history: list[ChatMessage]
    text_query: str
    data_query: str
    data_process_plan: str
    process_steps_plan: list[dict[str, Any]]
    pending_chart_params: dict[str, Any] | None
    process_repair_attempted: bool
    worker_step: dict[str, Any] | None
    node_flags: NodeEnableFlags
    report_mode: bool
    session_id: str

    knowledge_chunks: Annotated[list[ScoredChunk], _merge_knowledge_chunks]
    meta_hits: Annotated[list[ScoredMetaRecord], _merge_meta_hits]
    data_file_paths: Annotated[list[str], _merge_unique_paths]
    retrieval_round: int
    max_retrieval_rounds: int
    need_more_retrieval: bool
    retrieval_from_reporter: bool
    supplemental_retrieve_knowledge: bool
    supplemental_retrieve_data: bool
    after_reporter_retrieval_goto: str

    pending_tool: PendingToolCall | None
    plan_done: bool
    plan_step: int
    max_plan_tool_steps: int
    plan_steps: Annotated[list[PlanStepResult], operator.add]
    plan_context: dict[str, Any]

    process_result: dict[str, Any]
    process_done: bool
    process_step: Annotated[int, _merge_max_int]
    max_process_tool_steps: int
    data_tool_steps: Annotated[list[DataToolStepResult], operator.add]
    file_previews: Annotated[dict[str, FilePreviewEntry], _merge_file_previews]
    processed_data: Annotated[list[ProcessedDataRef], operator.add]
    processed_data_refs: Annotated[list[str], operator.add]
    chart_artifacts: Annotated[list[ChartArtifact], operator.add]

    report_done: bool
    report_step: int
    max_report_tool_steps: int
    report_steps: Annotated[list[ReportStepResult], operator.add]
    report_context: dict[str, Any]

    file_cache: list[SessionFileCache]
    nodes_traversed: Annotated[list[str], operator.add]
    final_answer: str
    report_artifact: ReportArtifact | None
    status: AgentStatus
    query_rejected: bool


@dataclass
class AgentRuntime:
    settings: Settings
    llm: LLMClient
    text_retriever: EnsembleRetriever
    meta_retriever: MetaEnsembleRetriever
    plan_registry: ToolRegistry
    data_registry: ToolRegistry
    report_registry: ToolRegistry
    llm_planner: LLMClient | None = None
    llm_data: LLMClient | None = None
    llm_reporter: LLMClient | None = None

    def llm_for_planner(self) -> LLMClient:
        return self.llm_planner or self.llm

    def llm_for_data(self) -> LLMClient:
        return self.llm_data or self.llm

    def llm_for_reporter(self) -> LLMClient:
        return self.llm_reporter or self.llm
