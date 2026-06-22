"""Conditional edge routing for LangGraph agent."""

from __future__ import annotations

from langgraph.types import Send

from app.core.agent.nodes._helpers import is_one_shot_mode, process_entry_node
from app.core.agent.state import AgentState
from app.schemas.structured import PendingToolCall


def _pending_phase(state: AgentState) -> str | None:
    pending = state.get("pending_tool")
    if isinstance(pending, PendingToolCall):
        return pending.phase
    return None


def _after_process_target(state: AgentState) -> str:
    flags = state.get("node_flags")
    if state.get("pending_chart_params") and flags and flags.enable_chart:
        return "chart_node"
    return "reporter"


def route_after_plan_done(state: AgentState) -> str:
    flags = state.get("node_flags")
    if flags and (flags.enable_knowledge_retrieve or flags.enable_data_retrieve):
        return "retriever"
    if flags and flags.enable_process:
        return process_entry_node(state)
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
    goto = str(state.get("after_reporter_retrieval_goto") or "").strip()
    if goto == "reporter":
        return "reporter"
    if goto in ("data_processor", "process_planner"):
        return process_entry_node(state) if goto == "data_processor" else "process_planner"
    flags = state.get("node_flags")
    if flags and flags.enable_process:
        return process_entry_node(state)
    return "reporter"


def route_after_process_planner(state: AgentState) -> str:
    if state.get("process_done"):
        return "reporter"
    return "process_fanout"


def route_process_fanout(state: AgentState) -> list[Send] | str:
    steps = state.get("process_steps_plan") or []
    pandas_steps = [s for s in steps if str(s.get("tool") or "") == "pandas_execute"]
    if len(pandas_steps) >= 2:
        return [Send("process_worker", {"worker_step": s}) for s in pandas_steps]
    return "process_executor"


def route_after_process_executor(state: AgentState) -> str:
    if state.get("process_done"):
        return _after_process_target(state)
    return "reporter"


def route_after_process_fanin(state: AgentState) -> str:
    return _after_process_target(state)


def route_after_chart_node(state: AgentState) -> str:
    return "reporter"


def route_after_data_processor(state: AgentState) -> str:
    if _pending_phase(state) == "data":
        return "data_tool"
    if state.get("process_done"):
        return "reporter"
    return "data_processor"


def route_after_data_tool(state: AgentState) -> str:
    if state.get("process_done"):
        return "reporter"
    return "data_processor"


def route_after_reporter(state: AgentState) -> str:
    if state.get("need_more_retrieval"):
        return "retriever"
    if _pending_phase(state) == "report":
        return "report_tool"
    return "reporter_end"


def route_after_report_tool(state: AgentState) -> str:
    return "reporter"
