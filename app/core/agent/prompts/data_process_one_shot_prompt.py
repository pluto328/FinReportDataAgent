"""One-shot data process plan prompt — single LLM call, no replan."""

from __future__ import annotations

import json
from typing import Any

from app.core.agent.nodes._helpers import (
    data_tool_catalog,
    format_file_previews_for_prompt,
    format_meta_columns_for_prompt,
    normalize_data_tool_params,
    user_require_text,
)
from app.core.agent.state import AgentRuntime, AgentState


def _format_retrieved_files(paths: list[str]) -> str:
    if not paths:
        return "（暂无检索到的数据文件）"
    lines = [f"{i}. {p}" for i, p in enumerate(paths, 1)]
    return f"共 {len(paths)} 个:\n" + "\n".join(lines)


def build_data_process_one_shot_prompt(state: AgentState, runtime: AgentRuntime) -> str:
    file_paths = list(state.get("data_file_paths") or [])
    flags = state.get("node_flags")
    enable_chart = bool(flags and flags.enable_chart)
    previews_text = format_file_previews_for_prompt(state.get("file_previews"))
    meta_text = format_meta_columns_for_prompt(state.get("meta_hits") or [])
    tool_catalog = data_tool_catalog(runtime)
    chart_rule = "可含 1 个 make_chart 步骤。" if enable_chart else "不要 make_chart 步骤。"

    return (
        "你是数据处理规划器。根据用户需求与已预览数据，一次性输出完整处理步骤 JSON，禁止 replan。\n"
        "输出格式（不要其它文字）：\n"
        '{"steps":[{"tool":"pandas_execute|sql_execute|data_filter","params":{...}}, ...]}\n'
        f"规则：\n"
        f"- 已自动加载原始文件预览，禁止 preview_read。\n"
        f"- 尽量 1 次 pandas_execute 完成筛选/排序/TopN；双产出题（如龙虎榜+负债）可用 2 个 pandas_execute。\n"
        f"- {chart_rule}\n"
        f"- pandas 多文件变量 df/df2/…；禁止 import、注释、pd.read_*。\n"
        f"- params 须含 artifact_name、artifact_description。\n"
        f"可用工具说明:\n{tool_catalog}\n"
        f"用户需求:{user_require_text(state)}\n"
        f"检索到的数据文件:\n{_format_retrieved_files(file_paths)}\n"
        f"元数据摘要:\n{meta_text}\n"
        f"已预览数据(CSV):\n{previews_text}\n"
    )


def build_data_process_repair_prompt(
    state: AgentState,
    runtime: AgentRuntime,
    *,
    failed_step: dict[str, Any],
    error: str,
) -> str:
    base = build_data_process_one_shot_prompt(state, runtime)
    return (
        f"{base}\n"
        f"上一步失败: tool={failed_step.get('tool')} error={error}\n"
        f"请仅输出修正后的 steps JSON（可只含失败步骤的修正版）。\n"
    )


def parse_one_shot_steps(raw: str, *, file_paths: list[str]) -> list[dict[str, Any]]:
    from app.core.agent.nodes._helpers import parse_llm_json

    data = parse_llm_json(raw)
    steps = data.get("steps") or []
    if not isinstance(steps, list):
        return []
    out: list[dict[str, Any]] = []
    for item in steps:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool") or item.get("tool_name") or "").strip()
        if tool == "preview_read":
            continue
        params = dict(item.get("params") or {})
        params = normalize_data_tool_params(params, file_paths=file_paths, tool_name=tool)
        out.append({"tool": tool, "params": params})
    return out
