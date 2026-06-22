"""data_processor node — ReAct + dynamic replan for data tools."""

from __future__ import annotations

import asyncio
from typing import Any

from pathlib import Path

from app.core.agent.events import emit_progress_waiting, invoke_llm_decision
from app.core.agent.nodes._debug_runtime import print_node_result, sample_state, stub_runtime
from app.core.agent.nodes._helpers import append_node, summarize_data_tool_steps
from app.core.agent.nodes._node_log import node_logger
from app.core.agent.prompts.data_processor_prompt import (
    build_data_processor_prompt,
    parse_data_processor_response,
)
from app.core.agent.state import AgentRuntime, AgentState
from app.schemas.structured import PendingToolCall, ProcessedDataRef


async def data_processor_node(state: AgentState, runtime: AgentRuntime) -> dict:
    log = node_logger(runtime.settings, "data_processor")
    flags = state.get("node_flags")
    if not flags or not flags.enable_process:
        log.start(enabled=False)
        log.info("数据处理未启用，跳过")
        log.end(process_done=True, skipped=True)
        return {**append_node(state, "data_processor"), "process_done": True, "after_reporter_retrieval_goto": "", "pending_tool": None}

    paths = state.get("data_file_paths") or []
    if not paths:
        log.start(enabled=True, file_paths=[])
        log.info("无可用数据文件，跳过处理")
        log.end(process_done=True, reason="no_files")
        return {**append_node(state, "data_processor"), "process_done": True, "after_reporter_retrieval_goto": "", "pending_tool": None}

    primary_path = paths[0]
    cache = state.get("file_cache") or []
    max_steps = state.get("max_process_tool_steps", runtime.settings.max_process_tool_steps)
    current_step = state.get("process_step", 0)
    prior_steps = state.get("data_tool_steps") or []
    log.start(
        file_paths=paths,
        process_step=current_step,
        max_steps=max_steps,
        prior_tool_steps=len(prior_steps),
    )

    for item in cache:
        if item.file_path == primary_path and item.processed_artifact_path:
            ref = ProcessedDataRef(
                path=item.processed_artifact_path,
                preview="(cached)",
                mode="tool",
                source_file=primary_path,
            )
            log.info("命中缓存的处理结果", path=item.processed_artifact_path)
            log.end(process_done=True, reused=True, path=item.processed_artifact_path)
            return {
                "process_result": {"reused": True, "path": item.processed_artifact_path, "preview": ref.preview},
                "processed_data": [ref],
                "processed_data_refs": [item.processed_artifact_path],
                "process_done": True,
                "after_reporter_retrieval_goto": "",
                "pending_tool": None,
                **append_node(state, "data_processor"),
            }

    force_no_tool = current_step >= max_steps
    if force_no_tool:
        log.info("数据处理步数已达上限，禁止 call_tool")

    prompt = build_data_processor_prompt(state, runtime, force_no_tool=force_no_tool)
    log.debug("调用 LLM 决定数据处理步骤", process_step=current_step)
    await emit_progress_waiting("正在规划数据处理", active=True)
    raw = await invoke_llm_decision(
        runtime.llm_for_data(), prompt, phase="data_processor", emit_thinking=False
    )
    await emit_progress_waiting(active=False)
    parsed = parse_data_processor_response(
        raw,
        file_paths=paths,
        current_step=current_step,
    )

    action = parsed["action"]
    tool_name = parsed["tool_name"]
    params = parsed["params"]
    new_plan = parsed["data_process_plan"]
    plan_empty = not str(state.get("data_process_plan") or "").strip()
    previews = state.get("file_previews") or {}
    paths = state.get("data_file_paths") or []
    previews_loaded = bool(paths) and all(Path(p).name in previews for p in paths if p)

    if plan_empty and not prior_steps and not force_no_tool:
        if previews_loaded:
            log.info("预览已自动加载，跳过 replan，等待 LLM 直接 call_tool")
        else:
            log.info("data_process_plan 为空，强制改为 replan")
            action = "replan"
            tool_name = ""
            params = {}

    if force_no_tool and action in ("call_tool", "replan"):
        log.info("数据处理步数已达上限，忽略 {}", action)
        action = "done"
        tool_name = ""
        params = {}

    log.info("LLM 数据处理决策", action=action, tool_name=tool_name if action != "done" else "")

    if action == "done":
        summary = summarize_data_tool_steps(prior_steps)
        out: dict[str, Any] = {
            "process_done": True,
            "after_reporter_retrieval_goto": "",
            "pending_tool": None,
            "process_result": summary,
            **append_node(state, "data_processor"),
        }
        log.end(process_done=True, tool_steps=len(prior_steps))
        return out

    if action == "replan":
        out_replan: dict[str, Any] = {
            "process_done": False,
            "pending_tool": None,
            **append_node(state, "data_processor"),
        }
        if new_plan:
            out_replan["data_process_plan"] = new_plan
        log.info("LLM 请求 replan", updated_plan=bool(new_plan))
        log.end(process_done=False, next_node="data_processor", action="replan")
        return out_replan

    out_extra: dict[str, Any] = {
        "process_done": False,
        "pending_tool": PendingToolCall(
            phase="data",
            tool_name=tool_name,
            params=params,
            file_path=str(params.get("file_path") or primary_path),
        ),
        **append_node(state, "data_processor"),
    }

    log.info("触发 data tool 调用", tool_name=tool_name, params=params)
    log.end(process_done=False, next_node="data_tool", tool_name=tool_name)
    return out_extra


async def debug_data_processor_node() -> None:
    state = sample_state(data_file_paths=["data/raw_structured/sample.csv"])
    runtime = stub_runtime()
    result = await data_processor_node(state, runtime)
    print_node_result("data_processor_node", result)


if __name__ == "__main__":
    asyncio.run(debug_data_processor_node())
