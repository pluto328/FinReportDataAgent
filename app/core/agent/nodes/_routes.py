"""Conditional edge routing for LangGraph agent."""

from __future__ import annotations

from app.core.agent.state import AgentState
from app.schemas.structured import PendingToolCall


def _pending_phase(state: AgentState) -> str | None:
    pending = state.get("pending_tool")
    if isinstance(pending, PendingToolCall):
        return pending.phase
    return None


def route_after_plan_done(state: AgentState) -> str:
    flags = state.get("node_flags")
    if flags and (flags.enable_knowledge_retrieve or flags.enable_data_retrieve):
        return "retriever"
    if flags and flags.enable_process:
        return "data_processor"
    return "reporter"


def route_after_planner(state: AgentState) -> str:
    if state.get("query_rejected"):
        return "planner_end"
    phase = _pending_phase(state)
    if phase == "plan":
        return "planning_tool"
    if not state.get("plan_done"):
        return "planner"
    return route_after_plan_done(state)


def route_after_planning_tool(state: AgentState) -> str:
    return "planner"


def route_after_retriever(state: AgentState) -> str:
    flags = state.get("node_flags")
    if flags and flags.enable_process:
        return "data_processor"
    return "reporter"


def route_after_data_processor(state: AgentState) -> str:
    if _pending_phase(state) == "data":
        return "data_tool"
    return "reporter"


def route_after_data_tool(state: AgentState) -> str:
    return "data_processor"


def route_after_reporter(state: AgentState) -> str:
    if state.get("need_more_retrieval"):
        return "retriever"
    if _pending_phase(state) == "report":
        return "report_tool"
    return "reporter_end"


def route_after_report_tool(state: AgentState) -> str:
    return "reporter"
