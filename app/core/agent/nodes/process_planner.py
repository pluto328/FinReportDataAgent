"""One-shot process planner — single LLM call produces steps JSON."""

from __future__ import annotations

import asyncio

from app.core.agent.events import emit_progress_waiting, invoke_llm_decision
from app.core.agent.nodes._helpers import append_node
from app.core.agent.nodes._node_log import node_logger
from app.core.agent.prompts.data_process_one_shot_prompt import (
    build_data_process_one_shot_prompt,
    parse_one_shot_steps,
)
from app.core.agent.state import AgentRuntime, AgentState


async def process_planner_node(state: AgentState, runtime: AgentRuntime) -> dict:
    log = node_logger(runtime.settings, "process_planner")
    flags = state.get("node_flags")
    paths = state.get("data_file_paths") or []

    if not flags or not flags.enable_process:
        log.info("数据处理未启用，跳过")
        return {**append_node(state, "process_planner"), "process_done": True, "pending_tool": None}

    if not paths:
        log.info("无数据文件，跳过处理")
        return {**append_node(state, "process_planner"), "process_done": True, "pending_tool": None}

    log.start(file_count=len(paths))
    prompt = build_data_process_one_shot_prompt(state, runtime)
    await emit_progress_waiting("正在规划数据处理", active=True)
    raw = ""
    try:
        raw = await asyncio.wait_for(
            invoke_llm_decision(
                runtime.llm_for_data(),
                prompt,
                phase="process_planner",
                emit_thinking=False,
            ),
            timeout=runtime.settings.llm_decision_timeout_sec,
        )
    except asyncio.TimeoutError:
        log.info("规划 LLM 超时", timeout_sec=runtime.settings.llm_decision_timeout_sec)
    finally:
        await emit_progress_waiting(active=False)

    steps = parse_one_shot_steps(raw, file_paths=list(paths))
    if not steps:
        log.info("one-shot 未解析到步骤，标记完成")
        return {
            **append_node(state, "process_planner"),
            "process_steps_plan": [],
            "process_done": True,
            "pending_tool": None,
        }

    log.info("one-shot 计划步骤", count=len(steps), tools=[s.get("tool") for s in steps])
    return {
        **append_node(state, "process_planner"),
        "process_steps_plan": steps,
        "process_repair_attempted": False,
        "process_done": False,
        "pending_tool": None,
    }
