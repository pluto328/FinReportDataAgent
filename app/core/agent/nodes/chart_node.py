"""Chart node — deterministic make_chart after data processing."""

from __future__ import annotations

from typing import Any

from app.core.agent.nodes._helpers import append_node
from app.core.agent.nodes._node_log import node_logger
from app.core.agent.nodes.data_tool_runner import execute_data_tool
from app.core.agent.state import AgentRuntime, AgentState


async def chart_node(state: AgentState, runtime: AgentRuntime) -> dict:
    log = node_logger(runtime.settings, "chart_node")
    flags = state.get("node_flags")
    params = dict(state.get("pending_chart_params") or {})

    if not flags or not flags.enable_chart or not params:
        log.info("无需绘图，跳过")
        return {
            **append_node(state, "chart_node"),
            "pending_chart_params": None,
            "process_done": True,
        }

    step_no = state.get("process_step", 0) + 1
    log.start(step=step_no, chart_type=params.get("chart_type"))
    patch = await execute_data_tool(
        state,
        runtime,
        tool_name="make_chart",
        params=params,
        step_no=step_no,
    )
    out: dict[str, Any] = {
        **append_node(state, "chart_node"),
        **patch,
        "pending_chart_params": None,
        "process_done": bool(patch.get("success")),
        "pending_tool": None,
    }
    log.end(success=patch.get("success"))
    return out
