"""Planner LLM prompt builder."""

from __future__ import annotations

from app.core.agent.nodes._helpers import (
    apply_plan_flags,
    data_tool_catalog,
    extract_history_context,
    format_plan_history,
    plan_tool_catalog,
)
from app.core.agent.prompts.planner_domain_knowledge import PLANNER_DOMAIN_KNOWLEDGE
from app.core.agent.state import AgentRuntime, AgentState
from app.core.session.process_artifact_store import format_intermediate_catalog, get_session_catalog


def _entry_validation_rules(has_history: bool) -> str:
    history_note = (
        "含「刚才/之前/继续/上述/同样」等明显追问上文时，即使表述简短也不要 reject。"
        if has_history
        else ""
    )
    return (
        "入口校验：若用户输入为寒暄闲聊、意义不明、或与金融/研报/企业数据/投资/财报无关，"
        '则 action=reject 并填 reject_reason（chitchat|unclear|non_finance）。'
        f"{history_note}\n"
    )


def build_planner_prompt(state: AgentState, runtime: AgentRuntime) -> str:
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
    catalog_text = format_intermediate_catalog(
        get_session_catalog(session_id, runtime.settings)
    )

    loaded_hint = prior_history.get("context_text", "") if history_loaded else "（尚未加载）"

    reject_schema = (
        '"action":"reject|call_tool|done","reject_reason":"chitchat|unclear|non_finance",'
        if is_entry_pass
        else '"action":"call_tool|done",'
    )
    entry_rules = _entry_validation_rules(has_history) if is_entry_pass else ""
    operation_rules ="""若会话历史数据（路径:描述）已包含所需中间结果，可跳过数据检索与数据处理，直接图表生成或报告生成。
    若涉及数据计算且历史数据不包含所需数据，则进行数据检索，
    若需要阅读相关报告才能得出结论，则进行知识检索，
    若所得结果涉及对比、排名、趋势、且数据量有限且适合用二维图表示则进行图表生成。
    若问题复杂/需要分析/涉及多维度/需要综合判断/给出建议/结论/则进行报告生成。
    若用户输入明确流程，则按照用户需求决定"""
    return (
        f"用户问题:{query}\n"
        "你是任务规划器"
        "你需要：1.判断用户问题是否有效，规则为"
        f"{entry_rules}"
        "2.判断问题含有刚才/之前/继续/上述/同样等追问上文时，调用工具load_history_context。action = call_tool。否则action = done。"
        "3.分析问题，有历史内容时结合历史问题分析，判断具体需求，确定以下操作是否进行:知识检索、数据检索、数据处理、图表生成、报告生成。并填写enable_*，填true或false。规则为"
        f"{operation_rules}\n"
        "4.如果需要进行文本检索，提取需求，扩写，尽量检索出更多有效信息，并填写text_query。\n"
        "5.如果需要数据处理，分点标号、列出处理步骤，大致描述为取数据、数据处理、数据计算、保存结果、是否画图。填写data_process_plan。\n"
        "6.根据数据处理要求填写data_query,要求使其尽量精准检索到相关数据文件"
        "仅输出 JSON，不要输出任何其他内容：\n"
        "{" + reject_schema +
        '"text_query":"","data_query":"","data_process_plan":"",'
        '"enable_knowledge_retrieve":false,"enable_data_retrieve":false,'
        '"enable_process":false,"enable_chart":false,"enable_report":false}\n'
        f"已调用plantool：({plan_step}/{max_steps}):\n{plan_history}\n"
        f"已加载历史上下文:\n{loaded_hint}\n"
        f"会话历史数据（路径:描述）:\n{catalog_text}\n"
        "规划时可参考以下常识理解用户问题：\n"
        f"{PLANNER_DOMAIN_KNOWLEDGE}\n"
    )


def parse_planner_response(
    raw: str,
    *,
    query: str,
    report_mode: bool,
    plan_steps: list,
    history_loaded: bool,
) -> dict:
    from app.core.agent.nodes._helpers import parse_llm_json
    from app.schemas.structured import NodeEnableFlags

    action = "done"
    tool_name = "load_history_context"
    params: dict = {}
    text_q, data_q, flags = query, query, NodeEnableFlags()
    data_process_plan = ""
    try:
        data = parse_llm_json(raw)
        action = data.get("action", "done")
        tool_name = data.get("tool_name") or data.get("method", tool_name)
        params = dict(data.get("params") or {})
        text_q, data_q, flags, data_process_plan = apply_plan_flags(
            data, query, report_mode
        )
        if action not in ("call_tool", "done", "reject"):
            action = "done" if plan_steps else "call_tool"
        return {
            "action": action,
            "tool_name": tool_name,
            "params": params,
            "text_query": text_q,
            "data_query": data_q,
            "node_flags": flags,
            "data_process_plan": data_process_plan,
            "reject_reason": str(data.get("reject_reason", "unclear")),
        }
    except Exception:
        lowered = query.lower()
        if any(k in lowered for k in ("csv", "xlsx", "table", "数据", "统计", "图表")):
            flags.enable_data_retrieve = True
            flags.enable_process = True
            data_process_plan = "读取并汇总结构化数据"
        if not history_loaded and plan_steps == []:
            action = "call_tool"
            tool_name = "load_history_context"
            params = {}
    return {
        "action": action,
        "tool_name": tool_name,
        "params": params,
        "text_query": text_q,
        "data_query": data_q,
        "node_flags": flags,
        "data_process_plan": data_process_plan,
        "reject_reason": "unclear",
    }
