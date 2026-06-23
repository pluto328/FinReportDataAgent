"""One-shot data process plan prompt — single LLM call, no replan."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.core.agent.nodes._helpers import (
    data_tool_catalog,
    format_file_previews_for_prompt,
    format_meta_columns_for_prompt,
    normalize_data_tool_params,
    user_require_text,
)
from app.core.agent.state import AgentRuntime, AgentState
from app.core.tools.data.pandas_execute import PANDAS_MULTI_FILE_VAR_RULE

_FILTER_NOISE = re.compile(
    r"查一下|帮我|请|这支股票|的信息|给出|建议|和|风险|分析|怎么样|如何|什么|哪些|相关|资料|数据"
)


def _extract_filter_keyword(state: AgentState) -> str:
    text = (user_require_text(state) or state.get("user_query") or "").strip()
    cleaned = _FILTER_NOISE.sub(" ", text)
    token = "".join(cleaned.split())
    return token[:24]


def build_fallback_process_steps(state: AgentState, file_paths: list[str]) -> list[dict[str, Any]]:
    """Heuristic pandas steps when one-shot LLM plan fails or times out."""
    keyword = _extract_filter_keyword(state)
    steps: list[dict[str, Any]] = []
    for fp in file_paths[:2]:
        stem = Path(fp).stem
        safe_kw = keyword.replace("\\", "\\\\").replace('"', '\\"')
        if safe_kw:
            code = (
                f'kw = "{safe_kw}"\n'
                "mask = df.astype(str).apply("
                'lambda r: r.str.contains(kw, case=False, na=False).any(), axis=1)\n'
                "result = df[mask] if mask.any() else df.head(100)"
            )
            desc = f"{stem} 关键词「{keyword}」筛选（fallback）"
        else:
            code = "result = df.head(100)"
            desc = f"{stem} 前 100 行（fallback）"
        steps.append(
            {
                "tool": "pandas_execute",
                "params": {
                    "file_paths": [fp],
                    "code": code,
                    "artifact_name": f"{stem}_fallback.csv",
                    "artifact_description": desc,
                },
            }
        )
    out: list[dict[str, Any]] = []
    for item in steps:
        params = normalize_data_tool_params(
            dict(item.get("params") or {}),
            file_paths=file_paths,
            tool_name=str(item.get("tool") or ""),
        )
        out.append({"tool": item["tool"], "params": params})
    return out


def _format_retrieved_files(paths: list[str]) -> str:
    if not paths:
        return "（暂无检索到的数据文件）"
    lines = [f"{i}. {p}" for i, p in enumerate(paths, 1)]
    return f"共 {len(paths)} 个:\n" + "\n".join(lines)


def _format_pandas_var_map(file_paths: list[str]) -> str:
    from app.core.tools.data.pandas_execute import _pandas_df_name

    if not file_paths:
        return "（无文件，无法映射 DataFrame 变量）"
    lines = [
        f"- 第{i + 1}个文件 {Path(p).name} → 变量 `{_pandas_df_name(i)}`"
        for i, p in enumerate(file_paths)
    ]
    return "运行时已注入的 DataFrame 变量（代码中直接使用，勿 read）：\n" + "\n".join(lines)


def _format_failed_step_detail(failed_step: dict[str, Any], error: str) -> str:
    tool = str(failed_step.get("tool") or failed_step.get("tool_name") or "")
    params = dict(failed_step.get("params") or {})
    code = str(params.get("code") or "").strip()
    sql = str(params.get("sql") or "").strip()
    other = {k: v for k, v in params.items() if k not in ("code", "sql")}
    lines = [f"工具: {tool or '（未知）'}", f"错误信息: {str(error).strip() or '（无）'}"]
    if code:
        lines.append(f"失败 pandas 代码（完整，请据此修改）:\n{code}")
    if sql:
        lines.append(f"失败 SQL（完整，请据此修改）:\n{sql}")
    if other:
        lines.append(f"其它参数: {json.dumps(other, ensure_ascii=False)}")
    return "\n".join(lines)


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
        f"- {PANDAS_MULTI_FILE_VAR_RULE}\n"
        f"- SQL 多文件表名：src（第1个）、src1（第2个）、src2（第3个）…\n"
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
    params = dict(failed_step.get("params") or {})
    step_paths = [str(p) for p in (params.get("file_paths") or []) if p]
    file_paths = step_paths or list(state.get("data_file_paths") or [])
    meta_text = format_meta_columns_for_prompt(state.get("meta_hits") or [])
    previews_text = format_file_previews_for_prompt(state.get("file_previews"), display_rows=5)
    failed_detail = _format_failed_step_detail(failed_step, error)
    var_map = _format_pandas_var_map(file_paths)
    return (
        "你是数据处理修正器。上一步工具执行失败，请仅输出修正后的 steps JSON，不要其它文字。\n"
        '{"steps":[{"tool":"pandas_execute|sql_execute|data_filter","params":{...}}]}\n'
        "规则：\n"
        "- 必须根据下方「错误信息」与「失败代码」修正，不得原样重复未改动的代码。\n"
        "- 禁止 preview_read；params 须含 artifact_name、artifact_description。\n"
        f"- {PANDAS_MULTI_FILE_VAR_RULE}\n"
        "- SQL 多文件表名：src/src1/src2…\n"
        f"用户需求:{user_require_text(state)}\n"
        f"失败详情:\n{failed_detail}\n"
        f"{var_map}\n"
        f"检索到的数据文件:\n{_format_retrieved_files(file_paths)}\n"
        f"元数据列信息:\n{meta_text}\n"
        f"数据预览(节选):\n{previews_text}\n"
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
