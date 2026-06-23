"""Execute one data tool step (LangGraph Send worker for parallel pandas)."""

from __future__ import annotations

from typing import Any

from app.core.agent.nodes._helpers import append_node
from app.core.agent.nodes._node_log import node_logger
from app.core.agent.nodes.data_tool_runner import execute_data_tool
from app.core.agent.state import AgentRuntime, AgentState


async def process_worker_node(state: AgentState, runtime: AgentRuntime) -> dict:
    log = node_logger(runtime.settings, "process_worker")
    step = state.get("worker_step") or {}
    tool_name = str(step.get("tool") or "").strip()
    params = dict(step.get("params") or {})
    base_step = state.get("process_step", 0)
    worker_index = state.get("worker_index")
    step_no = base_step + int(worker_index) + 1 if worker_index is not None else base_step + 1

    log.start(tool_name=tool_name, step=step_no)
    if not tool_name:
        log.end(skipped=True)
        return {**append_node(state, "process_worker")}

    patch = await execute_data_tool(
        state,
        runtime,
        tool_name=tool_name,
        params=params,
        step_no=step_no,
    )
    log.end(tool_name=tool_name, success=patch.get("success"), step=step_no)
    # Parallel Send workers must not write process_result/process_step (LangGraph single-value keys).
    safe_keys = frozenset({
        "data_tool_steps",
        "processed_data_refs",
        "chart_artifacts",
        "file_previews",
    })
    out: dict[str, Any] = {
        **append_node(state, "process_worker"),
        **{k: v for k, v in patch.items() if k in safe_keys},
    }
    return out
