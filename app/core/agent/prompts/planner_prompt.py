"""Planner LLM prompt builder."""

from __future__ import annotations

from app.core.agent.nodes._helpers import (
    apply_plan_flags,
    data_tool_catalog,
    extract_history_context,
    format_plan_history,
    plan_tool_catalog,
)
from app.core.agent.state import AgentRuntime, AgentState


def _entry_validation_rules(has_history: bool) -> str:
    history_note = (
        "含「刚才/之前/继续/上述/同样」等明显追问上文时，即使表述简短也不要 reject。"
        if has_history
        else ""
    )
    return (
        "0. 入口校验：若用户输入为寒暄闲聊、意义不明、或与金融/研报/企业数据/投资/财报无关，"
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
    tool_catalog = plan_tool_catalog(runtime)
    data_catalog = data_tool_catalog(runtime)
    loaded_hint = prior_history.get("context_text", "") if history_loaded else "（尚未加载）"

    reject_schema = (
        '"action":"reject|call_tool|done","reject_reason":"chitchat|unclear|non_finance",'
        if is_entry_pass
        else '"action":"call_tool|done",'
    )
    entry_rules = _entry_validation_rules(has_history) if is_entry_pass else ""

    return (
        "你是任务规划器。先判断用户问题是否有效，再决定是否需要结合会话历史、规划工具或完成规划。输出 JSON：\n"
        "{" + reject_schema +
        '"tool_name":"load_history_context",'
        '"params":{},'
        '"text_query":"","data_query":"","data_process_description":"",'
        '"enable_knowledge_retrieve":true,"enable_data_retrieve":false,'
        '"enable_process":false,"enable_chart":false,"enable_report":false}\n'
        f"可用规划工具:\n{tool_catalog}\n"
        f"data-processor 可用工具:\n{data_catalog}\n"
        f"已执行规划步骤({plan_step}/{max_steps}):\n{plan_history}\n"
        f"已加载历史上下文:\n{loaded_hint}\n"
        f"当前问题:{query}\n"
        "规则：\n"
        f"{entry_rules}"
        "1. 先判断当前问题是否需要结合前文（指代、追问、承接上文、省略主语等）；"
        "若需要且尚未 load_history_context，则 action=call_tool、tool_name=load_history_context、params 为空。\n"
        "2. 已加载历史后，结合 history 与当前问题填写 text_query、data_query，再 action=done。\n"
        "3. 明显独立新话题且无需历史时，直接 action=done。\n"
        "4. 规划完成时 action=done 并填写 enable_*、query 与 data_process_description。"
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
    data_process_description = ""
    try:
        data = parse_llm_json(raw)
        action = data.get("action", "done")
        tool_name = data.get("tool_name") or data.get("method", tool_name)
        params = dict(data.get("params") or {})
        text_q, data_q, flags, data_process_description = apply_plan_flags(
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
            "data_process_description": data_process_description,
            "reject_reason": str(data.get("reject_reason", "unclear")),
        }
    except Exception:
        lowered = query.lower()
        if any(k in lowered for k in ("csv", "xlsx", "table", "数据", "统计", "图表")):
            flags.enable_data_retrieve = True
            flags.enable_process = True
            data_process_description = "读取并汇总结构化数据"
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
        "data_process_description": data_process_description,
        "reject_reason": "unclear",
    }
