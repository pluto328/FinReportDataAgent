"""Data processor LLM prompt builder."""

from __future__ import annotations

from app.core.agent.nodes._helpers import data_tool_catalog, format_data_tool_history
from app.core.agent.state import AgentRuntime, AgentState
from app.core.session.process_artifact_store import format_intermediate_catalog, get_session_catalog


def build_data_processor_prompt(state: AgentState, runtime: AgentRuntime) -> str:
    file_path = (state.get("data_file_paths") or [""])[0]
    current_step = state.get("process_step", 0)
    max_steps = state.get("max_process_tool_steps", runtime.settings.max_process_tool_steps)
    prior_steps = state.get("data_tool_steps") or []
    data_process_plan = state.get("data_process_plan", "")
    history_text = format_data_tool_history(prior_steps)
    session_id = state.get("session_id", "")
    catalog_text = format_intermediate_catalog(
        get_session_catalog(session_id, runtime.settings)
    )
    tool_catalog = data_tool_catalog(runtime)
    tool_rules = """第一步，结合处理需求，对相关文件进行预览（preview_read，不产生新文件，无需 artifact_description）。
    第二步，如果需要进行进一步数据处理，则调用工具计算，只计算不绘图。
    纯筛选数据，则调用 data_filter 工具。
    逻辑复杂、执行 code，则调用 pandas_execute（code 禁止 import、pd.read_*、任何注释 #，直接使用 df/pd/np，结果赋给 result 或写回 df）。
    需 SQL 查询，则调用 sql_execute（sql 禁止任何注释 -- 或 /* */，仅 SELECT 语句正文）。
    第三步，如果需要画图，则调用 make_chart 工具。
    凡会保存新文件的工具（data_filter、sql_execute、pandas_execute、make_chart），
    必须在 params 中填写 artifact_description：用简短中文说明该文件是什么，
    例如「负债榜前五名数据」「北向持股 Top10 柱状图」，便于后续节点识别中间数据。
    preview_read 与 action=done/replan 时不需要 artifact_description。"""

    return (
        f"用户问题:{state.get('user_query','')}\n"
        f"data_process_plan（planner 给出的数据处理计划）:\n{data_process_plan}\n"
        f"数据文件:{file_path}\n"
        f"已知数据记录（路径:描述）:\n{catalog_text}\n"
        f"工具调用历史({current_step}/{max_steps}):\n{history_text}\n"
        "请结合 data_process_plan 与工具调用历史决定下一步："
        "若已知数据已满足要求则 action=done；"
        "若需继续处理则 action=call_tool 并选择合适工具；"
        "若发现原计划不合理或需调整步骤，则 action=replan 并重新填写 data_process_plan（分点标号，含取数、处理、计算、保存、是否画图）。\n"
        "若最新步骤为 error，根据 error 调整工具入参重试；同一工具连续两次 error 则换工具重试。\n"
        "call_tool 且工具会产出文件时，params 必须含 artifact_description（见下方规则）。\n"
        "仅输出 JSON，不要输出任何其他内容："
        '{"action":"call_tool|done|replan",'
        '"tool_name":"",'
        '"params":{"file_path":"","artifact_description":""},'
        '"data_process_plan":""}\n'
        "示例（筛选并保存）："
        '{"action":"call_tool","tool_name":"pandas_execute",'
        '"params":{"file_path":"/abs/path/data.csv","code":"df=df[df[\'col\']>0]; result=df",'
        '"artifact_description":"负债榜前五名数据"},"data_process_plan":""}\n'
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
