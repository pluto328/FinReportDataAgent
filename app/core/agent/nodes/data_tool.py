"""data_tool node — execute data_registry tools."""

from __future__ import annotations

import asyncio

from app.core.agent.events import emit_tool_start
from app.core.agent.nodes._debug_runtime import print_node_result, sample_state, stub_runtime
from app.core.agent.nodes._helpers import append_node, is_data_process_plan_complete
from app.core.agent.nodes._node_log import node_logger
from app.core.agent.nodes.data_tool_runner import execute_data_tool
from app.core.agent.state import AgentRuntime, AgentState
from app.schemas.structured import PendingToolCall


async def data_tool_node(state: AgentState, runtime: AgentRuntime) -> dict:
    log = node_logger(runtime.settings, "data_tool")
    pending = state.get("pending_tool")
    if not pending or pending.phase != "data" or not pending.tool_name:
        log.start(plan_action="skip")
        log.info("无待执行 data tool，跳过")
        log.end(process_done=False)
        return {**append_node(state, "data_tool"), "process_done": False, "pending_tool": None}

    step_no = state.get("process_step", 0) + 1
    tool_name = pending.tool_name
    params = dict(pending.params)
    log.start(tool_name=tool_name, step=step_no, params=params)
    log.info("触发 data tool 调用", tool_name=tool_name)

    patch = await execute_data_tool(
        state,
        runtime,
        tool_name=tool_name,
        params=params,
        step_no=step_no,
    )

    out: dict = {
        **append_node(state, "data_tool"),
        **{k: v for k, v in patch.items() if k not in ("success", "error")},
        "pending_tool": None,
        "process_done": False,
    }

    if patch.get("success"):
        merged_state = {**state, **out}
        if is_data_process_plan_complete(merged_state):
            out["process_done"] = True
            log.info("计划步骤已全部完成，自动结束数据处理")
            next_node = "reporter"
        else:
            next_node = "data_processor"
    else:
        next_node = "data_processor"

    log.end(
        tool_name=tool_name,
        step=step_no,
        success=bool(patch.get("success")),
        next_node=next_node,
        process_done=out.get("process_done"),
    )
    return out


async def debug_data_tool_node() -> None:
    state = sample_state(
        pending_tool=PendingToolCall(
            phase="data",
            tool_name="preview_read",
            file_path="data/raw_structured/sample.csv",
            params={},
        ),
        data_file_paths=["data/raw_structured/sample.csv"],
    )
    runtime = stub_runtime()
    result = await data_tool_node(state, runtime)
    print_node_result("data_tool_node", result)


if __name__ == "__main__":
    asyncio.run(debug_data_tool_node())
