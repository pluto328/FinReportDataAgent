"""Planner LLM prompt builder."""

from __future__ import annotations

from app.core.agent.nodes._helpers import (
    apply_plan_flags,
    extract_history_context,
    format_plan_history,
    parse_user_require,
    plan_tool_catalog,
)
from app.core.agent.prompts.planner_domain_knowledge import PLANNER_DOMAIN_KNOWLEDGE
from app.core.agent.state import AgentRuntime, AgentState
from app.core.session.process_artifact_store import format_intermediate_catalog_for_agent, get_session_catalog


def _entry_validation_rules(has_history: bool) -> str:
    return (
        "入口校验：用户输入为闲聊、意义不明、或与金融/研报/企业数据/投资/财报无关，则 action=reject"
    )


def build_planner_prompt(
    state: AgentState,
    runtime: AgentRuntime,
    *,
    force_no_tool: bool = False,
) -> str:
    query = state.get("user_query", "")
    plan_steps = state.get("plan_steps") or []
    plan_step = state.get("plan_step", 0)
    max_steps = state.get("max_plan_tool_steps", runtime.settings.max_plan_tool_steps)
    prior_history = extract_history_context(plan_steps)
    history_loaded = bool(prior_history.get("context_text") or prior_history.get("history"))
    chat_hist = state.get("chat_history") or []
    has_history = bool(chat_hist) or history_loaded
    is_entry_pass = plan_step == 0 and not plan_steps

    plan_history = format_plan_history(plan_steps)
    session_id = state.get("session_id", "")
    catalog_text = format_intermediate_catalog_for_agent(
        get_session_catalog(session_id, runtime.settings)
    )
    loaded_hint = prior_history.get("context_text", "") if history_loaded else "（尚未加载）"

    force_line = ""
    if force_no_tool:
        force_line = f"已达最大 plan tool 步数({max_steps})，必须 action=done，禁止 call_tool。\n"
    entry_rules = _entry_validation_rules(has_history) if is_entry_pass else ""
    if force_no_tool:
        tool_catalog_block = "（已达 plan tool 步数上限，禁止 call_tool）"
    else:
        catalog = plan_tool_catalog(runtime)
        history_hint = ""
        if has_history and not history_loaded:
            history_hint = (
                "\n仅当含有刚才/上述/那么等上下文上下文强相关，或无法理解用户需求时调用"
                " action=call_tool，tool_name=load_history_context，params 留空。"
            )
        tool_catalog_block = f"{catalog}{history_hint}" if catalog else "（无）"
    operation_rules = """若会话历史数据（文件名:描述）已包含所需中间结果，可跳过数据检索与数据处理，直接图表生成或报告生成。
    若涉及数据计算且历史数据不包含所需数据，则进行数据检索，
    若需要阅读相关报告才能得出结论，则进行知识检索，
    若所得结果涉及对比、排名、趋势、且数据量有限且适合用二维图表示则进行图表生成。
    若问题复杂/需要分析/涉及多维度/需要综合判断/给出建议/结论/则进行报告生成。
    若用户输入明确流程，则按照用户需求决定。
"""

    return (
        "你是企业知识库咨询系统的流程规划器。需要理解用户真实需求，确定后续流程。\n"
        "按以下固定格式输出 JSON，不要输出任何其他内容。'#'后为填写规则：\n"
        '{"planning_thought":"",#必填，用 1-3 句中文描述规划思路，句式以「用户让我…」或「用户想咨询…」开头，接着写「我需要先…再…」（可继续「然后…」）\n'
        '"action":"",#必填，若下一步调用工具，则填：call_tool；若用户需求已明确，则填：done。'
        f"可调用 plan 工具:\n{tool_catalog_block}\n"
        f"{entry_rules}"
        '；否则填：done\n'
        '"user_require":"",#必填，推测的用户完整真实任务意图，不复述原话，表述更精确。\n'
        '"text_query":"",\n#仅当 enable_knowledge_retrieve 为 true 时填写（检索扩写词，填具体需求的金融强相关词，不得填语气词或关联词）；为 false 时 text_query 必须为空字符串。\n'
        '"data_query":"",\n#仅当 enable_data_retrieve 为 true 时填写（数据检索词，填具体需求的金融强相关词，不得填语气词或关联词）；为 false 时 data_query 必须为空字符串。\n'
        '"enable_knowledge_retrieve":"","#必填，是否需要知识检索，填true或false。\n'
        '"enable_data_retrieve":"","#必填，是否需要数据检索，填true或false。\n'
        '"enable_process":"","#必填，是否需要数据处理，填true或false。\n'
        '"enable_chart":"","#必填，是否需要图表生成，填true或false。\n'
        '"enable_report":"","#必填，是否需要报告生成，填true或false。\n'
        '"tool_name":"",#当action=call_tool时，填下一步调用工具的名称，否则为空字符串。\n'
        '"params":{},#当action=call_tool时，填下一步调用工具的参数，否则为空对象。\n'
        '}\n'
        "已知信息:\n"
        f"用户问题:{query}\n"
        f"已加载历史上下文:\n{loaded_hint}\n"
        f"{operation_rules}\n"
        f"{force_line}"
        f"已调用 plan tool（{plan_step}/{max_steps}）:\n{plan_history}\n"

        f"已知数据（文件名:描述）:\n{catalog_text}\n"
        "规划时可参考以下常识理解用户问题：\n"
        f"{PLANNER_DOMAIN_KNOWLEDGE}\n"
    )


def parse_planner_response(
    raw: str,
    *,
    query: str,
    report_mode: bool,
) -> dict:
    from app.core.agent.nodes._helpers import parse_llm_json
    from app.schemas.structured import NodeEnableFlags

    action = "done"
    tool_name = ""
    params: dict = {}
    text_q, data_q, flags = "", "", NodeEnableFlags()
    user_require = query
    reject_reason = "unclear"
    try:
        data = parse_llm_json(raw)
        action = str(data.get("action", "done") or "done")
        tool_name = str(data.get("tool_name") or data.get("method") or "").strip()
        params = dict(data.get("params") or {})
        text_q, data_q, flags = apply_plan_flags(data, query, report_mode)
        user_require = parse_user_require(data, fallback=query)
        reject_reason = str(data.get("reject_reason", "unclear"))
        if action not in ("call_tool", "done", "reject"):
            action = "done"
            tool_name = ""
            params = {}
        if action != "call_tool":
            tool_name = ""
            params = {}
    except Exception:
        action = "done"
        tool_name = ""
        params = {}
        text_q, data_q = "", ""
        flags = NodeEnableFlags()
        user_require = query
    return {
        "action": action,
        "tool_name": tool_name,
        "params": params,
        "text_query": text_q,
        "data_query": data_q,
        "node_flags": flags,
        "user_require": user_require,
        "reject_reason": reject_reason,
    }
