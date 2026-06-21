"""Data processor LLM prompt builder."""

from __future__ import annotations

from app.core.agent.nodes._helpers import data_tool_catalog, format_data_tool_history, normalize_data_tool_params
from app.core.agent.state import AgentRuntime, AgentState
from app.core.session.process_artifact_store import (
    format_intermediate_catalog_for_agent,
    get_session_catalog,
)


def _format_retrieved_files(paths: list[str]) -> str:
    if not paths:
        return "（暂无检索到的数据文件）"
    lines = [f"{i}. {p}" for i, p in enumerate(paths, 1)]
    return f"共 {len(paths)} 个:\n" + "\n".join(lines)


def build_data_processor_prompt(
    state: AgentState,
    runtime: AgentRuntime,
    *,
    force_no_tool: bool = False,
) -> str:
    file_paths = list(state.get("data_file_paths") or [])
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
    retrieved_files_text = _format_retrieved_files(file_paths)
    tool_rules = """
    preview_read只对所检索的原始数据使用，不能对中间数据使用，中间产物的预览从工具调用历史中查看。
    sql_execute 与 pandas_execute 使用 file_paths（绝对路径列表），可同时处理多个检索到的原始数据文件；需要多表关联或合并时，应一次传入全部相关 file_paths。
    SQL 多文件时表名：src（第1个）、src2（第2个）、src3（第3个）…；pandas 多文件时变量名：df（第1个）、df2（第2个）、df3（第3个）…。
    生成pandas代码或sql代码需要尽量一步到位，不要多次连续调用pandas_execute或连续调用sql_execute。
    逻辑复杂，则生成pandas代码后使用pandas_execute（code 禁止 import、pd.read_*、任何注释 #，直接使用 df/df2/…/pd/np，结果赋给 result 或写回 df）。
    需 SQL 查询，则调用 sql_execute（sql 禁止任何注释 -- 或 /* */，仅 SELECT 语句正文）。
    """

    force_line = ""
    if force_no_tool:
        force_line = f"已达最大 data tool 步数({max_steps})，必须 action=done，禁止 call_tool 与 replan。\n"

    return (
        f"用户问题:{state.get('user_query','')}\n"
        f"data_process_plan:\n{data_process_plan}\n"
        f"检索到的数据文件:\n{retrieved_files_text}\n"
        f"已知数据记录（文件名:描述）:\n{catalog_text}\n"
        f"工具调用历史({current_step}/{max_steps}):\n{history_text}\n"
        f"{force_line}"
        "根据工具调用历史，若最新步骤返回为 error，根据 error 调整工具入参重试；同一工具连续两次 error 则换工具重试。\n"
        "请结合 data_process_plan 与工具调用历史决定下一步："
        "若已知数据已满足要求则 action=done；"
        "若需继续处理则 action=call_tool 并选择合适工具，并填写tool_name，和params，可调用data工具和调用规则见下文。"
        "若当前所执行步骤和所得数据已无法满足需求，需重新规划数据处理计划，则 action=replan 并填写 data_process_plan。"
        "仅输出 JSON，不要输出任何其他内容："
        '{"action":"call_tool|done|replan",'
        '"tool_name":"",'
        '"params":{"file_path":"","file_paths":[],"artifact_name":"","artifact_description":""},'
        '"data_process_plan":""}\n'
        "params 说明：preview_read/data_filter/make_chart 用 file_path（单个绝对路径）；"
        "sql_execute/pandas_execute 用 file_paths（绝对路径列表，可从上方检索到的数据文件中选取一个或多个）。\n"
        f"可调用 data 工具:\n{tool_catalog}\n"
        "调用工具规则为：\n"
        f"{tool_rules}\n"
    )


def parse_data_processor_response(
    raw: str,
    *,
    file_paths: list[str],
    current_step: int,
) -> dict:
    from app.core.agent.nodes._helpers import parse_llm_json

    default_path = file_paths[0] if file_paths else ""
    action = "call_tool"
    tool_name = "preview_read"
    params: dict = {"file_path": default_path, "file_paths": list(file_paths)}
    data_process_plan = ""

    try:
        data = parse_llm_json(raw)
        action = data.get("action", action)
        tool_name = data.get("tool_name", tool_name)
        params.update(data.get("params") or {})
        params = normalize_data_tool_params(params, file_paths=file_paths)
        data_process_plan = str(
            data.get("data_process_plan", "")
            or data.get("dataprocessplan", "")
            or data.get("data_process_description", "")
            or ""
        )
        if action not in ("call_tool", "done", "replan"):
            action = "done"
    except Exception:
        action = "done"
    return {
        "action": action,
        "tool_name": tool_name,
        "params": params,
        "data_process_plan": data_process_plan,
    }
