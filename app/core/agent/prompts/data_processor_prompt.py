"""Data processor LLM prompt builder."""

from __future__ import annotations

import json

from app.core.agent.nodes._helpers import (
    data_tool_catalog,
    format_data_tool_history,
)
from app.core.agent.state import AgentRuntime, AgentState
from app.core.session.process_artifact_store import format_intermediate_catalog


def build_data_processor_prompt(state: AgentState, runtime: AgentRuntime) -> str:
    file_path = (state.get("data_file_paths") or [""])[0]
    current_step = state.get("process_step", 0)
    max_steps = state.get("max_process_tool_steps", runtime.settings.max_process_tool_steps)
    prior_steps = state.get("data_tool_steps") or []
    plan_ctx = state.get("plan_context") or {}
    history_block = ""
    if isinstance(plan_ctx, dict):
        hc = plan_ctx.get("history_context") or {}
        history_block = hc.get("context_text", "") if isinstance(hc, dict) else ""
    data_desc = state.get("data_process_description", "")
    history_text = format_data_tool_history(prior_steps)
    tool_catalog = data_tool_catalog(runtime)
    catalog = state.get("intermediate_data_catalog") or {}
    catalog_text = format_intermediate_catalog(catalog)
    last_error = prior_steps[-1].error if prior_steps else ""

    error_rule = ""
    if last_error:
        error_rule = (
            f"上一步工具执行失败：{last_error}。"
            "优先 action=replan 或 call_tool 修正参数/换工具，不要直接 done。\n"
        )

    return (
        f"用户问题:{state.get('user_query','')}\n"
        f"历史上下文:\n{history_block}\n"
        f"规划上下文:{json.dumps(plan_ctx, ensure_ascii=False)[:600]}\n"
        f"高层处理目标（可随 replan 更新）:{data_desc}\n"
        f"数据文件:{file_path}\n"
        f"全局中间数据记录（路径:描述）:\n{catalog_text}\n"
        f"已执行数据处理步骤({current_step}/{max_steps}):\n{history_text}\n"
        "根据上述观测结果决定下一步。输出 JSON："
        '{"action":"call_tool|done|replan","tool_name":"preview_read",'
        '"params":{},"intermediate_data":{},"data_process_description":"",'
        '"secondary_text_query":"","secondary_data_query":""}\n'
        f"可用工具:\n{tool_catalog}\n"
        "规则：\n"
        f"{error_rule}"
        "每次输出必须包含完整的 intermediate_data 对象：所有已知中间数据路径及其处理描述。\n"
        "call_tool 时在 params 中为新生成的中间数据填写 artifact_description。\n"
        "信息已足够则 action=done；否则 action=call_tool 并指定 tool_name 与 params。\n"
        "上一步失败或策略需调整时 action=replan，填写更新后的 data_process_description 与下一步 tool_name/params。\n"
        "需要 SQL 时用 sql_execute（params 含 sql）；复杂变换用 pandas_execute（params 含 code）；"
        "需要图表时用 make_chart（params 含 chart_type/x_axis/y_axis/title）。"
    )


def parse_data_processor_response(
    raw: str,
    *,
    file_path: str,
    current_step: int,
    prior_catalog: dict[str, str] | None = None,
) -> dict:
    from app.core.agent.nodes._helpers import parse_llm_json

    action = "call_tool"
    tool_name = "preview_read"
    params: dict = {"file_path": file_path}
    data_process_description = ""
    secondary_text = ""
    secondary_data = ""
    intermediate_data = dict(prior_catalog or {})
    try:
        data = parse_llm_json(raw)
        action = data.get("action", action)
        tool_name = data.get("tool_name", tool_name)
        params.update(data.get("params") or {})
        params.setdefault("file_path", file_path)
        data_process_description = str(data.get("data_process_description", "") or "")
        secondary_text = str(data.get("secondary_text_query", "") or "")
        secondary_data = str(data.get("secondary_data_query", "") or "")
        llm_catalog = data.get("intermediate_data")
        if isinstance(llm_catalog, dict):
            for k, v in llm_catalog.items():
                if k:
                    intermediate_data[str(k)] = str(v)
        if action not in ("call_tool", "done", "replan"):
            action = "done" if current_step > 0 else "call_tool"
    except Exception:
        if current_step > 0:
            action = "done"
    return {
        "action": action,
        "tool_name": tool_name,
        "params": params,
        "data_process_description": data_process_description,
        "secondary_text_query": secondary_text,
        "secondary_data_query": secondary_data,
        "intermediate_data": intermediate_data,
    }
