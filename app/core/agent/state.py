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


class AgentState(TypedDict, total=False):
    user_query: str
    user_require: str
    chat_history: list[ChatMessage]
    text_query: str
    data_query: str
    data_process_plan: str
    node_flags: NodeEnableFlags
    report_mode: bool
    session_id: str

    knowledge_chunks: Annotated[list[ScoredChunk], operator.add]
    meta_hits: Annotated[list[ScoredMetaRecord], operator.add]
    data_file_paths: Annotated[list[str], operator.add]
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
    process_step: int
    max_process_tool_steps: int
    data_tool_steps: Annotated[list[DataToolStepResult], operator.add]
    file_previews: dict[str, FilePreviewEntry]
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
