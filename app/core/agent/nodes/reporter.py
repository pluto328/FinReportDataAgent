"""reporter node — integrate context, optional report tools, write Markdown (ReAct)."""

from __future__ import annotations

import asyncio

from app.core.agent.events import invoke_llm_decision
from app.core.agent.nodes._debug_runtime import print_node_result, sample_state, stub_runtime
from app.core.agent.nodes._helpers import (
    append_node,
    summarize_data_tool_steps,
    summarize_plan_steps,
    summarize_report_steps,
)
from app.core.agent.nodes._node_log import node_logger
from app.core.agent.prompts.reporter_prompt import build_reporter_prompt, parse_reporter_response
from app.core.agent.state import AgentRuntime, AgentState
from app.core.agent.query_guard import INSUFFICIENT_DATA_MESSAGE
from app.core.session.history_store import append_session_turn
from app.core.session.process_artifact_store import get_session_catalog, resolve_catalog_path
from app.schemas.session import SessionTurnRecord
from app.schemas.structured import PendingToolCall


def _retrieval_insufficient(state: AgentState) -> bool:
    flags = state.get("node_flags")
    if not flags:
        return False
    chunks = state.get("knowledge_chunks") or []
    meta = state.get("meta_hits") or []
    k_en = flags.enable_knowledge_retrieve
    d_en = flags.enable_data_retrieve
    if not k_en and not d_en:
        return False
    k_fail = k_en and not chunks
    d_fail = d_en and not meta
    if k_en and d_en:
        return k_fail and d_fail
    return k_fail or d_fail


def _append_supplemental_queries(
    report_context: dict,
    *,
    text_q: str = "",
    data_q: str = "",
) -> dict:
    ctx = dict(report_context)
    if text_q:
        prev = list(ctx.get("supplemental_text_queries") or [])
        prev.append(text_q)
        ctx["supplemental_text_queries"] = prev
    if data_q:
        prev = list(ctx.get("supplemental_data_queries") or [])
        prev.append(data_q)
        ctx["supplemental_data_queries"] = prev
    return ctx


async def _fallback_summary(llm, body: str, user_query: str) -> str:
    if not body.strip():
        return ""
    prompt = (
        f"用户问题:{user_query}\n"
        f"回答内容:{body[:4000]}\n"
        "请用2-3句中文概括上述内容的核心结论与关键数据。只输出摘要正文。"
    )
    raw = await llm.ainvoke(prompt)
    text = (raw or "").strip()[:800]
    from app.core.agent.llm_capture import record_llm_call

    record_llm_call(phase="reporter", purpose="fallback_summary", prompt=prompt, output=text)
    return text


async def _persist_turn(
    state: AgentState,
    runtime: AgentRuntime,
    *,
    answer: str,
    answer_summary: str,
    markdown_path: str = "",
) -> None:
    session_id = state.get("session_id", "default")
    record = SessionTurnRecord(
        turn_id=0,
        question=state.get("user_query", ""),
        answer_summary=answer_summary,
        report_mode=False,
        markdown_path=markdown_path,
    )
    append_session_turn(record, session_id, runtime.settings)


async def _finalize_from_parsed(
    state: AgentState,
    runtime: AgentRuntime,
    parsed: dict,
    *,
    chunks: list,
    meta: list,
    report_steps: list,
    charts: list,
    log,
) -> dict:
    summary = parsed.get("summary") or ""
    final_status = state.get("status", "ok")
    answer = parsed.get("answer") or parsed.get("report") or ""
    if not summary:
        summary = await _fallback_summary(runtime.llm, answer, state.get("user_query", ""))
    await _persist_turn(state, runtime, answer=answer, answer_summary=summary)
    log.info("回答生成成功", answer_len=len(answer))
    log.end(report_done=True, status=final_status, answer_len=len(answer))
    return {
        "final_answer": answer,
        "status": final_status,
        "need_more_retrieval": False,
        "report_done": True,
        "pending_tool": None,
        "report_context": {**summarize_report_steps(report_steps), "answer_summary": summary},
        **append_node(state, "reporter"),
    }


async def reporter_node(state: AgentState, runtime: AgentRuntime) -> dict:
    log = node_logger(runtime.settings, "reporter")
    query = state.get("user_query", "")
    chunks = state.get("knowledge_chunks") or []
    meta = state.get("meta_hits") or []
    data_steps = state.get("data_tool_steps") or []
    plan_context = state.get("plan_context") or summarize_plan_steps(state.get("plan_steps") or [])
    process_result = state.get("process_result") or summarize_data_tool_steps(data_steps)
    report_steps = state.get("report_steps") or []
    report_step = state.get("report_step", 0)
    max_report_steps = state.get("max_report_tool_steps", runtime.settings.max_report_tool_steps)
    report_context = dict(state.get("report_context") or {})
    retrieval_round = state.get("retrieval_round", 0)
    max_rounds = state.get("max_retrieval_rounds", runtime.settings.max_retrieval_rounds)
    charts = state.get("chart_artifacts") or []

    log.start(
        session_id=state.get("session_id", ""),
        query=query,
        knowledge_chunks=len(chunks),
        meta_hits=len(meta),
        report_step=report_step,
        retrieval_round=retrieval_round,
    )

    if not chunks and not meta and not process_result and not plan_context and not report_context:
        if _retrieval_insufficient(state):
            log.info("检索已启用但未达到 rerank 阈值", min_rerank_score=runtime.settings.min_rerank_score)
            log.end(status="not_found", retrieval_insufficient=True)
            return {
                "final_answer": INSUFFICIENT_DATA_MESSAGE,
                "status": "not_found",
                "report_done": True,
                "pending_tool": None,
                **append_node(state, "reporter"),
            }
        if retrieval_round + 1 < max_rounds:
            log.info("上下文不足，触发补充检索", round=retrieval_round + 1)
            log.end(need_more_retrieval=True, status="retry")
            return {
                "retrieval_round": retrieval_round + 1,
                "need_more_retrieval": True,
                **append_node(state, "reporter"),
            }
        log.info("无法查询到目标数据")
        log.end(status="not_found", report_done=True)
        return {
            "final_answer": "无法查询到目标数据",
            "status": "not_found",
            "report_done": True,
            "pending_tool": None,
            **append_node(state, "reporter"),
        }

    force_done = report_step >= max_report_steps or bool(state.get("report_done"))
    stream_field = "answer"

    parsed: dict = {}
    action = "done"
    tool_name = ""
    params: dict = {}
    text_q = ""
    data_q = ""

    for attempt in range(3):
        prompt = build_reporter_prompt(state, runtime, force_done=force_done)
        if attempt > 0:
            log.debug("重复 read_data_file 后重问 LLM", attempt=attempt + 1)
        else:
            log.debug("调用 LLM", report_step=report_step, force_done=force_done, stream_field=stream_field)
        raw = await invoke_llm_decision(
            runtime.llm,
            prompt,
            phase="reporter",
            stream_field=stream_field if attempt == 0 else None,
            stream_as="answer",
            emit_thinking=False,
        )
        parsed = parse_reporter_response(raw)
        action = parsed["action"]
        tool_name = parsed["tool_name"]
        params = parsed["params"]
        text_q = parsed["text_query"]
        data_q = parsed["data_query"]
        if force_done and action in ("call_tool", "retrieve_text", "retrieve_data"):
            log.info("报告步数已达上限，忽略 {}", action)
            action = "done"
            tool_name = ""
            params = {}
            text_q = ""
            data_q = ""
        log.info("LLM 报告决策", action=action, tool_name=tool_name if action == "call_tool" else "")

        if force_done or action != "call_tool" or tool_name != "read_data_file":
            break

        session_id = state.get("session_id", "")
        extra_paths = list(state.get("data_file_paths") or [])
        extra_paths.extend(state.get("processed_data_refs") or [])
        raw_path = str(params.get("path", ""))
        resolved = resolve_catalog_path(
            session_id, raw_path, runtime.settings, extra_paths=extra_paths
        )
        if resolved:
            params["path"] = resolved
        loaded_paths = set(report_context.get("loaded_paths") or [])
        if report_context.get("loaded_path"):
            loaded_paths.add(str(report_context["loaded_path"]))
        if resolved and resolved in loaded_paths:
            log.info("跳过重复 read_data_file，直接生成回答", path=resolved)
            action = "done"
            tool_name = ""
            params = {}
            break
        break

    if not force_done and action == "retrieve_text" and text_q and retrieval_round + 1 < max_rounds:
        log.info("触发补充文本检索", text_query=text_q)
        log.end(need_more_retrieval=True, next_node="retriever")
        return {
            "text_query": text_q,
            "retrieval_round": retrieval_round + 1,
            "need_more_retrieval": True,
            "retrieval_from_reporter": True,
            "supplemental_retrieve_knowledge": True,
            "supplemental_retrieve_data": False,
            "report_context": _append_supplemental_queries(report_context, text_q=text_q),
            "pending_tool": None,
            **append_node(state, "reporter"),
        }

    if not force_done and action == "retrieve_data" and data_q and retrieval_round + 1 < max_rounds:
        log.info("触发补充数据检索", data_query=data_q)
        log.end(need_more_retrieval=True, next_node="retriever")
        return {
            "data_query": data_q,
            "retrieval_round": retrieval_round + 1,
            "need_more_retrieval": True,
            "retrieval_from_reporter": True,
            "supplemental_retrieve_knowledge": False,
            "supplemental_retrieve_data": True,
            "report_context": _append_supplemental_queries(report_context, data_q=data_q),
            "pending_tool": None,
            **append_node(state, "reporter"),
        }

    if not force_done and action == "call_tool" and tool_name:
        if tool_name == "read_data_file":
            session_id = state.get("session_id", "")
            extra_paths = list(state.get("data_file_paths") or [])
            extra_paths.extend(state.get("processed_data_refs") or [])
            raw_path = str(params.get("path", ""))
            if not params.get("path"):
                resolved = resolve_catalog_path(
                    session_id, raw_path, runtime.settings, extra_paths=extra_paths
                )
                if resolved:
                    params["path"] = resolved
                else:
                    catalog = get_session_catalog(session_id, runtime.settings)
                    if catalog:
                        params["path"] = next(iter(catalog))
        log.info("触发 report tool 调用", tool_name=tool_name, params=params)
        log.end(report_done=False, next_node="report_tool", tool_name=tool_name)
        return {
            "report_done": False,
            "pending_tool": PendingToolCall(phase="report", tool_name=tool_name, params=params),
            **append_node(state, "reporter"),
        }

    log.info("生成最终输出")
    return await _finalize_from_parsed(
        state,
        runtime,
        parsed,
        chunks=chunks,
        meta=meta,
        report_steps=report_steps,
        charts=charts,
        log=log,
    )


async def debug_reporter_node() -> None:
    state = sample_state(plan_context={"step_count": 1, "latest": {"result": "2026-01-01"}})
    runtime = stub_runtime()
    result = await reporter_node(state, runtime)
    print_node_result("reporter_node", result)


if __name__ == "__main__":
    asyncio.run(debug_reporter_node())
