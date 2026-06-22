"""planner node — LLM task planning and NodeEnableFlags (ReAct)."""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

from app.core.agent.events import invoke_llm_decision
from app.core.agent.nodes._debug_runtime import print_node_result, sample_state, stub_runtime
from app.core.agent.nodes._helpers import (
    append_node,
    extract_history_context,
    summarize_plan_steps,
)
from app.core.agent.nodes._node_log import node_logger
from app.core.agent.prompts.planner_prompt import build_planner_prompt, parse_planner_response
from app.core.agent.query_guard import (
    GUIDANCE_MESSAGE,
    is_empty_query,
    looks_meaningless_by_rule,
    normalize_query,
)
from app.core.agent.state import AgentRuntime, AgentState
from app.schemas.structured import NodeEnableFlags, PendingToolCall


def _flags_line(flags: NodeEnableFlags | None) -> str:
    if not flags:
        return "all_off"
    return (
        f"knowledge={flags.enable_knowledge_retrieve} "
        f"data={flags.enable_data_retrieve} "
        f"process={flags.enable_process} "
        f"chart={flags.enable_chart} "
        f"report={flags.enable_report}"
    )


def _apply_history_to_queries(
    out: dict[str, Any],
    plan_steps: list,
    query: str,
) -> None:
    history_ctx = extract_history_context(plan_steps)
    if not history_ctx.get("context_text") and not history_ctx.get("history"):
        return
    plan_ctx = dict(out.get("plan_context") or {})
    if not isinstance(plan_ctx, dict):
        plan_ctx = {}
    plan_ctx["history_context"] = history_ctx
    out["plan_context"] = plan_ctx


def _reject_response(state: AgentState, query: str, reason: str) -> dict[str, Any]:
    sid = state.get("session_id") or uuid4().hex
    return {
        "user_query": query,
        "query_rejected": True,
        "plan_done": True,
        "report_done": True,
        "final_answer": GUIDANCE_MESSAGE,
        "status": "rejected",
        "session_id": sid,
        "pending_tool": None,
        **append_node(state, "planner"),
    }


async def planner_node(state: AgentState, runtime: AgentRuntime) -> dict:
    log = node_logger(runtime.settings, "planner")
    query = normalize_query(state.get("user_query", ""))
    plan_steps = state.get("plan_steps") or []
    plan_step = state.get("plan_step", 0)
    max_steps = state.get("max_plan_tool_steps", runtime.settings.max_plan_tool_steps)
    is_first = plan_step == 0 and not plan_steps
    prior_history = extract_history_context(plan_steps)
    history_loaded = bool(prior_history.get("context_text") or prior_history.get("history"))
    is_entry_pass = plan_step == 0 and not plan_steps

    log.start(
        session_id=state.get("session_id", ""),
        query=query,
        plan_step=plan_step,
        max_steps=max_steps,
        history_loaded=history_loaded,
    )
    log.info("正在规划链路开关与 query", user_query=query)

    if is_entry_pass:
        if is_empty_query(query):
            log.info("输入为空，直接结束", reason="empty")
            out = _reject_response(state, query, "empty")
            log.end(query_rejected=True, reason="empty", next_node="END")
            return out
        if looks_meaningless_by_rule(query):
            log.info("输入无意义（规则），直接结束", reason="unclear")
            out = _reject_response(state, query, "unclear")
            log.end(query_rejected=True, reason="unclear", next_node="END")
            return out

    force_no_tool = plan_step >= max_steps
    if force_no_tool:
        log.info("规划步数已达上限，禁止 call_tool")

    prompt = build_planner_prompt(state, runtime, force_no_tool=force_no_tool)
    log.debug("调用 LLM 规划（含入口校验）", plan_step=plan_step, entry_pass=is_entry_pass)
    raw = await invoke_llm_decision(
        runtime.llm, prompt, phase="planner", stream_field="planning_thought"
    )

    parsed = parse_planner_response(
        raw,
        query=query,
        report_mode=bool(state.get("report_mode")),
    )
    action = parsed["action"]
    tool_name = parsed["tool_name"]
    params = parsed["params"]
    text_q = parsed["text_query"]
    data_q = parsed["data_query"]
    flags = parsed["node_flags"]
    user_require = parsed["user_require"]

    if force_no_tool and action == "call_tool":
        log.info("规划步数已达上限，忽略 call_tool")
        action = "done"
        tool_name = ""
        params = {}

    if action == "reject" and is_entry_pass:
        reason = str(parsed.get("reject_reason", "unclear"))
        log.info("LLM 判定输入无效，直接结束", reason=reason)
        out = _reject_response(state, query, reason)
        log.end(query_rejected=True, reason=reason, next_node="END")
        return out

    log.info(
        "LLM 规划解析完成",
        action=action,
        tool_name=tool_name if action == "call_tool" else "",
        enables=_flags_line(flags),
    )

    if action == "done" and flags:
        log.info(
            "本次规划链路",
            enables=_flags_line(flags),
            text_query=text_q,
            data_query=data_q,
        )

    out: dict[str, Any] = {
        "user_query": query,
        "user_require": user_require,
        "text_query": text_q,
        "data_query": data_q,
        "node_flags": flags,
        **append_node(state, "planner"),
    }

    if is_first:
        out.update(
            {
                "retrieval_round": 0,
                "max_retrieval_rounds": runtime.settings.max_retrieval_rounds,
                "max_plan_tool_steps": runtime.settings.max_plan_tool_steps,
                "max_process_tool_steps": runtime.settings.max_process_tool_steps,
                "max_report_tool_steps": runtime.settings.max_report_tool_steps,
                "process_step": 0,
                "process_done": False,
                "report_step": 0,
                "report_done": False,
                "session_id": state.get("session_id") or uuid4().hex,
            }
        )
        log.info("初始化 session 与步数上限", session_id=out["session_id"])

    if action == "call_tool" and tool_name:
        log.info("触发 plan tool 调用", method=tool_name, params=params)
        out.update(
            {
                "plan_done": False,
                "pending_tool": PendingToolCall(phase="plan", tool_name=tool_name, params=params),
            }
        )
        log.end(plan_done=False, next_node="planning_tool", method=tool_name)
        return out

    out.update(
        {
            "plan_done": True,
            "pending_tool": None,
            "plan_context": summarize_plan_steps(plan_steps),
        }
    )
    _apply_history_to_queries(out, plan_steps, query)
    log.info("生成 query 成功", text_query=out.get("text_query"), data_query=out.get("data_query"))
    log.end(plan_done=True, enables=_flags_line(out.get("node_flags")))
    return out


async def debug_planner_node() -> None:
    state = sample_state()
    runtime = stub_runtime()
    result = await planner_node(state, runtime)
    print_node_result("planner_node", result)


if __name__ == "__main__":
    asyncio.run(debug_planner_node())
