"""Execute planned data tools sequentially or in parallel."""

from __future__ import annotations

from typing import Any

from app.core.agent.events import emit_progress_waiting, invoke_llm_decision
from app.core.agent.nodes._helpers import append_node
from app.core.agent.nodes._node_log import node_logger
from app.core.agent.nodes.data_tool_runner import execute_data_tool
from app.core.agent.prompts.data_process_one_shot_prompt import (
    build_data_process_repair_prompt,
    parse_one_shot_steps,
)
from app.core.agent.state import AgentRuntime, AgentState


def _split_steps(steps: list[dict[str, Any]], *, enable_chart: bool) -> tuple[list, list, list]:
    pandas_steps: list[dict] = []
    chart_steps: list[dict] = []
    other_steps: list[dict] = []
    for s in steps:
        tool = str(s.get("tool") or "")
        if tool == "make_chart":
            chart_steps.append(s)
        elif tool == "pandas_execute":
            pandas_steps.append(s)
        elif tool:
            other_steps.append(s)
    if not enable_chart:
        chart_steps = []
    return pandas_steps, other_steps, chart_steps


def _merge_patches(base: dict[str, Any], patch: dict[str, Any]) -> None:
    for key, val in patch.items():
        if key in ("success", "error"):
            continue
        if key in ("data_tool_steps", "processed_data", "processed_data_refs", "chart_artifacts"):
            base.setdefault(key, [])
            if isinstance(val, list):
                base[key].extend(val)
        else:
            base[key] = val


async def _run_steps(
    state: AgentState,
    runtime: AgentRuntime,
    steps: list[dict[str, Any]],
    *,
    start_step: int,
    parallel: bool = False,
) -> tuple[dict[str, Any], str, dict | None]:
    import asyncio

    merged: dict[str, Any] = {}
    work_state: dict[str, Any] = dict(state)
    failed_step: dict | None = None
    error = ""

    async def _one(step: dict, step_no: int) -> tuple[dict, dict | None, str]:
        tool = str(step.get("tool") or "")
        params = dict(step.get("params") or {})
        patch = await execute_data_tool(
            work_state,  # type: ignore[arg-type]
            runtime,
            tool_name=tool,
            params=params,
            step_no=step_no,
        )
        if patch.get("data_tool_steps"):
            work_state["data_tool_steps"] = [
                *(work_state.get("data_tool_steps") or []),
                *patch["data_tool_steps"],
            ]
        if patch.get("file_previews"):
            work_state["file_previews"] = patch["file_previews"]
        if patch.get("processed_data_refs"):
            work_state["processed_data_refs"] = [
                *(work_state.get("processed_data_refs") or []),
                *patch.get("processed_data_refs", []),
            ]
        if not patch.get("success"):
            return patch, step, str(patch.get("error") or "unknown")
        return patch, None, ""

    if parallel and len(steps) > 1:
        results = await asyncio.gather(
            *[_one(step, start_step + i) for i, step in enumerate(steps)]
        )
        for patch, fail, err in results:
            _merge_patches(merged, patch)
            if fail:
                failed_step = fail
                error = err
    else:
        step_no = start_step
        for step in steps:
            step_no += 1
            patch, fail, err = await _one(step, step_no)
            _merge_patches(merged, patch)
            if fail:
                failed_step = fail
                error = err
                break

    return merged, error, failed_step


async def process_executor_node(state: AgentState, runtime: AgentRuntime) -> dict:
    log = node_logger(runtime.settings, "process_executor")
    steps = list(state.get("process_steps_plan") or [])
    flags = state.get("node_flags")
    enable_chart = bool(flags and flags.enable_chart)
    start_step = state.get("process_step", 0)

    log.start(planned_steps=len(steps), start_step=start_step)
    if not steps:
        return {**append_node(state, "process_executor"), "process_done": True, "pending_tool": None}

    pandas_steps, other_steps, chart_steps = _split_steps(steps, enable_chart=enable_chart)
    merged: dict[str, Any] = {}
    failed_step: dict | None = None
    error = ""

    if len(pandas_steps) >= 2:
        p_patch, error, failed_step = await _run_steps(
            state, runtime, pandas_steps, start_step=start_step, parallel=True
        )
        _merge_patches(merged, p_patch)
        if not failed_step and other_steps:
            work = {**state, **merged}
            o_patch, error2, failed_step = await _run_steps(
                work, runtime, other_steps, start_step=merged.get("process_step", start_step), parallel=False
            )
            _merge_patches(merged, o_patch)
            if failed_step:
                error = error2
    else:
        all_exec = pandas_steps + other_steps
        patch, error, failed_step = await _run_steps(
            state, runtime, all_exec, start_step=start_step, parallel=False
        )
        _merge_patches(merged, patch)

    out: dict[str, Any] = {
        **append_node(state, "process_executor"),
        **merged,
        "pending_tool": None,
    }

    if failed_step and not state.get("process_repair_attempted"):
        log.info("步骤失败，尝试 repair", error=error[:120])
        repair_prompt = build_data_process_repair_prompt(
            {**state, **merged},
            runtime,
            failed_step=failed_step,
            error=error,
        )
        await emit_progress_waiting("正在修正数据处理", active=True)
        raw = await invoke_llm_decision(
            runtime.llm_for_data(),
            repair_prompt,
            phase="process_planner",
            purpose="repair",
            emit_thinking=False,
        )
        await emit_progress_waiting(active=False)
        repair_steps = parse_one_shot_steps(raw, file_paths=list(state.get("data_file_paths") or []))
        if repair_steps:
            work = {**state, **merged}
            r_patch, _, _ = await _run_steps(
                work,
                runtime,
                repair_steps,
                start_step=merged.get("process_step", start_step),
                parallel=False,
            )
            _merge_patches(merged, r_patch)
            out.update(merged)
            out["process_repair_attempted"] = True

    if chart_steps and enable_chart and not failed_step:
        out["pending_chart_params"] = dict(chart_steps[0].get("params") or {})
        out["process_done"] = False
    else:
        out["pending_chart_params"] = None
        out["process_done"] = True

    log.end(process_done=out.get("process_done"), has_chart=bool(out.get("pending_chart_params")))
    return out


async def process_fanin_node(state: AgentState, runtime: AgentRuntime) -> dict:
    """After parallel workers: run remaining non-pandas steps and defer chart."""
    log = node_logger(runtime.settings, "process_fanin")
    steps = list(state.get("process_steps_plan") or [])
    flags = state.get("node_flags")
    enable_chart = bool(flags and flags.enable_chart)
    remaining: list[dict] = []
    chart_params = None
    for s in steps:
        tool = str(s.get("tool") or "")
        if tool == "pandas_execute":
            continue
        if tool == "make_chart":
            if enable_chart:
                chart_params = dict(s.get("params") or {})
            continue
        remaining.append(s)

    log.start(remaining=len(remaining), has_chart=bool(chart_params))
    merged: dict[str, Any] = {**append_node(state, "process_fanin")}
    if remaining:
        patch, error, _ = await _run_steps(
            state,
            runtime,
            remaining,
            start_step=state.get("process_step", 0),
            parallel=False,
        )
        merged.update(patch)
        if error:
            merged["process_done"] = True
            merged["pending_chart_params"] = None
            log.end(error=error[:80])
            return merged

    if chart_params:
        merged["pending_chart_params"] = chart_params
        merged["process_done"] = False
    else:
        merged["pending_chart_params"] = None
        merged["process_done"] = True

    log.end(process_done=merged.get("process_done"))
    return merged
