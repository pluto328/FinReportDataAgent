"""Data processor LLM prompt builder."""

from __future__ import annotations

from app.core.agent.nodes._helpers import data_tool_catalog, format_data_tool_history
from app.core.agent.state import AgentRuntime, AgentState
from app.core.session.process_artifact_store import (
    format_intermediate_catalog_for_agent,
    get_session_catalog,
    resolve_catalog_path,
)


def build_data_processor_prompt(state: AgentState, runtime: AgentRuntime) -> str:
    file_path = (state.get("data_file_paths") or [""])[0]
    current_step = state.get("process_step", 0)
    max_steps = state.get("max_process_tool_steps", runtime.settings.max_process_tool_steps)
    prior_steps = state.get("data_tool_steps") or []
    data_process_plan = state.get("data_process_plan", "")
    history_text = format_data_tool_history(prior_steps)
    session_id = state.get("session_id", "")
    catalog_text = format_intermediate_catalog_for_agent(
        get_session_catalog(session_id, runtime.settings)
    )
    tool_catalog = data_tool_catalog(runtime)
    tool_rules = """
    preview_read只对所检索的原始数据使用，不能对中间数据使用，中间产物的预览从工具调用历史中查看。
    逻辑复杂，则生成pandas代码后使用pandas_execute（code 禁止 import、pd.read_*、任何注释 #，直接使用 df/pd/np，结果赋给 result 或写回 df）。
    需 SQL 查询，则调用 sql_execute（sql 禁止任何注释 -- 或 /* */，仅 SELECT 语句正文）。
    """

    return (
        f"用户问题:{state.get('user_query','')}\n"
        f"data_process_plan:\n{data_process_plan}\n"
        f"数据文件:{file_path}\n"
        f"已知数据记录（文件名:描述）:\n{catalog_text}\n"
        f"工具调用历史({current_step}/{max_steps}):\n{history_text}\n"
        "根据工具调用历史，若最新步骤返回为 error，根据 error 调整工具入参重试；同一工具连续两次 error 则换工具重试。\n"
        "请结合 data_process_plan 与工具调用历史决定下一步："
        "若已知数据已满足要求则 action=done；"
        "若需继续处理则 action=call_tool 并选择合适工具；"
        "仅输出 JSON，不要输出任何其他内容："
        '{"action":"call_tool|done|replan",'
        '"tool_name":"",'
        '"params":{"file_path":"","artifact_name":"","artifact_description":""},'
        '"data_process_plan":""}\n'

        f"可调用 data 工具:\n{tool_catalog}\n"
        "调用工具规则为：\n"
        f"{tool_rules}\n"
    )


def parse_data_processor_response(
    raw: str,
    *,
    file_path: str,
    current_step: int,
) -> dict:
    from app.core.agent.nodes._helpers import parse_llm_json

    action = "call_tool"
    tool_name = "preview_read"
    params: dict = {"file_path": file_path}
    data_process_plan = ""

    try:
        data = parse_llm_json(raw)
        action = data.get("action", action)
        tool_name = data.get("tool_name", tool_name)
        params.update(data.get("params") or {})
        params.setdefault("file_path", file_path)
        data_process_plan = str(
            data.get("data_process_plan", "")
            or data.get("dataprocessplan", "")
            or data.get("data_process_description", "")
            or ""
        )
        if action not in ("call_tool", "done", "replan"):
            action = "done" if current_step > 0 else "call_tool"
    except Exception:
        if current_step > 0:
            action = "done"
    return {
        "action": action,
        "tool_name": tool_name,
        "params": params,
        "data_process_plan": data_process_plan,
    }
