"""planning_tool node — execute plan_registry tools and return to planner."""

from __future__ import annotations

import asyncio

from app.core.agent.events import emit_tool_end, emit_tool_start
from app.core.agent.nodes._debug_runtime import print_node_result, sample_state, stub_runtime
from app.core.agent.nodes._helpers import append_node, summarize_plan_steps
from app.core.agent.nodes._node_log import node_logger
from app.core.agent.state import AgentRuntime, AgentState
from app.schemas.structured import PendingToolCall, PlanStepResult


async def planning_tool_node(state: AgentState, runtime: AgentRuntime) -> dict:
    log = node_logger(runtime.settings, "planning_tool")
    pending = state.get("pending_tool")
    if not pending or pending.phase != "plan" or not pending.tool_name:
        log.start(plan_action="skip")
        log.info("无待执行 plan tool，跳过")
        log.end(plan_done=True)
        return {**append_node(state, "planning_tool"), "plan_done": True, "pending_tool": None}

    step_no = state.get("plan_step", 0) + 1
    params = dict(pending.params)
    method = pending.tool_name
    log.start(method=method, step=step_no, params=params)

    if method == "load_history_context":
        params.setdefault("session_id", state.get("session_id", "default"))
        params.setdefault("settings", runtime.settings)
        history = state.get("chat_history") or []
        params.setdefault(
            "chat_history",
            [{"role": m.role, "content": m.content} for m in history],
        )

    tool = runtime.plan_registry.get(method)
    error = ""
    await emit_tool_start("plan", method)
    if tool:
        log.info("触发 plan tool 调用", method=method)
        try:
            result = await tool.run(**params)
            if isinstance(result, dict) and result.get("error"):
                error = str(result["error"])
                log.fail("plan tool 调用失败", method=method, error=error)
            else:
                log.info("plan tool 调用成功", method=method, result=result)
        except Exception as exc:
            result = {}
            error = str(exc)
            log.fail("plan tool 调用异常", method=method, error=error)
    else:
        result = {}
        error = f"plan tool not found: {method}"
        log.fail("plan tool 未找到", method=method)

    await emit_tool_end("plan", method, ok=not error, error=error)

    step = PlanStepResult(
        step=step_no,
        method=method,
        params={k: v for k, v in params.items() if k not in ("settings", "llm")},
        result=result if isinstance(result, dict) else {"result": result},
        error=error,
    )
    prior = state.get("plan_steps") or []
    all_steps = [*prior, step]
    log.end(method=method, step=step_no, success=not error, next_node="planner")
    return {
        "plan_steps": [step],
        "plan_step": step_no,
        "plan_context": summarize_plan_steps(all_steps),
        "plan_done": False,
        "pending_tool": None,
        **append_node(state, "planning_tool"),
    }


async def debug_planning_tool_node() -> None:
    state = sample_state(
        pending_tool=PendingToolCall(
            phase="plan",
            tool_name="load_history_context",
            params={},
        ),
    )
    runtime = stub_runtime()
    result = await planning_tool_node(state, runtime)
    print_node_result("planning_tool_node", result)


if __name__ == "__main__":
    asyncio.run(debug_planning_tool_node())
