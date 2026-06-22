"""Data processor LLM prompt builder."""

from __future__ import annotations

from app.core.agent.nodes._helpers import (
    data_tool_catalog,
    format_data_tool_history,
    format_file_previews_for_prompt,
    normalize_data_tool_params,
    user_require_text,
)
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
    previews_text = format_file_previews_for_prompt(state.get("file_previews"))
    session_id = state.get("session_id", "")
    catalog_text = format_intermediate_catalog_for_agent(
        get_session_catalog(session_id, runtime.settings)
    )
    tool_catalog = data_tool_catalog(runtime)
    retrieved_files_text = _format_retrieved_files(file_paths)
    tool_rules = """
    SQL 多文件时表名：src（第1个）、src2（第2个）、src3（第3个）…；pandas 多文件时变量名：df（第1个）、df2（第2个）、df3（第3个）…。
    生成pandas代码或sql代码需要尽量一步到位，不要多次连续调用pandas_execute或连续调用sql_execute。
    make_chart 只允许调用一次。
    原始文件预览已自动加载到「已预览数据」，禁止 call_tool preview_read。

    """

    force_line = ""
    if force_no_tool:
        force_line = f"已达最大 data tool 步数({max_steps})，必须 action=done，禁止 call_tool 与 replan。\n"

    plan_empty = not str(data_process_plan or "").strip()


    return (
        "你是数据处理规划+执行器。需要跟据现有文件和用户需求,调用工具进行数据处理。\n"
        "按以下固定格式输出 JSON，不要输出任何其他内容。'#'后为填写规则：\n"
        '{"action":"call_tool|done|replan",#必填，下一步调用工具，则填：call_tool；若已知数据已满足要求则填：done；若当前data_process_plan为空则填：replan。\n'
        '"tool_name":"",#当action=call_tool时，填下一步调用工具的名称。'
        f"可调用 data 工具说明:\n{tool_catalog}\n"
        "调用工具规则为：\n"
        f"{tool_rules}"
        '"params":{},#当action=call_tool时，填下一步调用工具的参数\n'
        '"data_process_plan":""#当action=replan时，根据用户真实需求，保证能满足需求的前提下，尽量使步骤最少。按顺序分点描述，每个点只描述一个独立操作，避免和/并之类的描述。pandas一次可以处理多张表，产生汇总表。\n'
        '}\n'
        "已知信息:\n"
        f"用户需求:{user_require_text(state)}\n"
        f"data_process_plan:\n{data_process_plan}\n"
        f"检索到的数据文件:\n{retrieved_files_text}\n"
        f"中间数据记录（文件名:描述）:\n{catalog_text}\n"
        f"已预览数据(文件名|描述|预览):\n{previews_text}\n"
        f"工具调用历史({current_step}/{max_steps}):\n{history_text}\n"
        f"{force_line}"
        "严格按照data_process_plan执行下一步。参照工具调用历史明确现在载哪一步。"
        "若最新步骤返回为 error，根据 error 调整工具入参重试；同一工具连续两次 error 则换工具重试。"
        "不得轻易启动replan，除非确定当前步骤有错误\n"

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
    params: dict = {"file_path": default_path}
    data_process_plan = ""

    try:
        data = parse_llm_json(raw)
        action = data.get("action", action)
        tool_name = str(data.get("tool_name", tool_name) or "").strip()
        raw_params = dict(data.get("params") or {})
        data_process_plan = str(
            data.get("data_process_plan", "")
            or data.get("dataprocessplan", "")
            or data.get("data_process_description", "")
            or ""
        )
        if action == "replan":
            tool_name = ""
            params = {}
        elif action == "done":
            tool_name = ""
            params = {}
        elif tool_name == "preview_read":
            params.update(raw_params)
            params = normalize_data_tool_params(params, file_paths=file_paths, tool_name=tool_name)
        else:
            params.update(raw_params)
            params = normalize_data_tool_params(params, file_paths=file_paths, tool_name=tool_name)
        if action not in ("call_tool", "done", "replan"):
            action = "done"
            tool_name = ""
            params = {}
    except Exception:
        action = "done"
        tool_name = ""
        params = {}
    return {
        "action": action,
        "tool_name": tool_name,
        "params": params,
        "data_process_plan": data_process_plan,
    }
