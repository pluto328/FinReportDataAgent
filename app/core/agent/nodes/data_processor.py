"""data_processor node — ReAct + dynamic replan for data tools."""

from __future__ import annotations

import asyncio
from typing import Any

from app.core.agent.events import invoke_llm_decision
from app.core.agent.nodes._debug_runtime import print_node_result, sample_state, stub_runtime
from app.core.agent.nodes._helpers import append_node, summarize_data_tool_steps
from app.core.agent.nodes._node_log import node_logger
from app.core.agent.prompts.data_processor_prompt import (
    build_data_processor_prompt,
    parse_data_processor_response,
)
from app.core.agent.state import AgentRuntime, AgentState
from app.core.session.process_artifact_store import merge_intermediate_catalog
from app.schemas.structured import PendingToolCall, ProcessedDataRef


async def data_processor_node(state: AgentState, runtime: AgentRuntime) -> dict:
    log = node_logger(runtime.settings, "data_processor")
    flags = state.get("node_flags")
    if not flags or not flags.enable_process:
        log.start(enabled=False)
        log.info("数据处理未启用，跳过")
        log.end(process_done=True, skipped=True)
        return {**append_node(state, "data_processor"), "process_done": True, "pending_tool": None}

    paths = state.get("data_file_paths") or []
    if not paths:
        log.start(enabled=True, file_paths=[])
        log.info("无可用数据文件，跳过处理")
        log.end(process_done=True, reason="no_files")
        return {**append_node(state, "data_processor"), "process_done": True, "pending_tool": None}

    file_path = paths[0]
    cache = state.get("file_cache") or []
    max_steps = state.get("max_process_tool_steps", runtime.settings.max_process_tool_steps)
    current_step = state.get("process_step", 0)
    prior_steps = state.get("data_tool_steps") or []
    log.start(
        file_path=file_path,
        process_step=current_step,
        max_steps=max_steps,
        prior_tool_steps=len(prior_steps),
    )

    for item in cache:
        if item.file_path == file_path and item.processed_artifact_path:
            ref = ProcessedDataRef(
                path=item.processed_artifact_path,
                preview="(cached)",
                mode="tool",
                source_file=file_path,
            )
            log.info("命中缓存的处理结果", path=item.processed_artifact_path)
            log.end(process_done=True, reused=True, path=item.processed_artifact_path)
            return {
                "process_result": {"reused": True, "path": item.processed_artifact_path, "preview": ref.preview},
                "processed_data": [ref],
                "processed_data_refs": [item.processed_artifact_path],
                "process_done": True,
                "pending_tool": None,
                **append_node(state, "data_processor"),
            }

    if current_step >= max_steps:
        summary = summarize_data_tool_steps(prior_steps)
        log.info("数据处理步数已达上限，结束", max_steps=max_steps)
        log.end(process_done=True, status="partial")
        return {
            "process_done": True,
            "pending_tool": None,
            "process_result": summary or {"error": "max data tool steps reached"},
            "status": "partial",
            **append_node(state, "data_processor"),
        }

    prompt = build_data_processor_prompt(state, runtime)
    log.debug("调用 LLM 决定数据处理步骤", process_step=current_step)
    raw = await invoke_llm_decision(runtime.llm, prompt, phase="data_processor")
    parsed = parse_data_processor_response(
        raw,
        file_path=file_path,
        current_step=current_step,
        prior_catalog=state.get("intermediate_data_catalog"),
    )

    action = parsed["action"]
    tool_name = parsed["tool_name"]
    params = parsed["params"]
    secondary_text = parsed["secondary_text_query"]
    secondary_data = parsed["secondary_data_query"]
    new_description = parsed["data_process_description"]
    intermediate_data = parsed["intermediate_data"]
    session_id = state.get("session_id", "default")

    if intermediate_data:
        merge_intermediate_catalog(session_id, intermediate_data, runtime.settings)

    log.info("LLM 数据处理决策", action=action, tool_name=tool_name if action != "done" else "")

    if action == "done":
        summary = summarize_data_tool_steps(prior_steps)
        out: dict[str, Any] = {
            "process_done": True,
            "pending_tool": None,
            "process_result": summary,
            "intermediate_data_catalog": intermediate_data,
            **append_node(state, "data_processor"),
        }
        if secondary_text:
            out["text_query"] = secondary_text
        if secondary_data:
            out["data_query"] = secondary_data
        log.end(process_done=True, tool_steps=len(prior_steps))
        return out

    out_extra: dict[str, Any] = {
        "process_done": False,
        "pending_tool": PendingToolCall(
            phase="data",
            tool_name=tool_name,
            params=params,
            file_path=file_path,
        ),
        "intermediate_data_catalog": intermediate_data,
        **append_node(state, "data_processor"),
    }
    if action == "replan" and new_description:
        out_extra["data_process_description"] = new_description
    if secondary_text:
        out_extra["text_query"] = secondary_text
    if secondary_data:
        out_extra["data_query"] = secondary_data

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
