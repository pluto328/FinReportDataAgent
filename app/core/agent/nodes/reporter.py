"""reporter node — integrate context, optional report tools, write Markdown (ReAct)."""

from __future__ import annotations

import asyncio

from app.core.agent.events import invoke_llm_decision, invoke_llm_report_output
from app.core.agent.nodes._debug_runtime import print_node_result, sample_state, stub_runtime
from app.core.agent.nodes._helpers import (
    append_node,
    extract_history_context,
    summarize_data_tool_steps,
    summarize_plan_steps,
    summarize_report_steps,
)
from app.core.agent.nodes._node_log import node_logger
from app.core.agent.prompts.reporter_prompt import (
    build_answer_prompt,
    build_reporter_decision_prompt,
    parse_reporter_decision_response,
)
from app.core.agent.state import AgentRuntime, AgentState
from app.core.agent.query_guard import INSUFFICIENT_DATA_MESSAGE
from app.core.session.history_store import append_session_turn
from app.core.session.process_artifact_store import get_session_catalog
from app.schemas.session import SessionTurnRecord
from app.schemas.structured import PendingToolCall, ReportArtifact


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


async def _fallback_summary(llm, report: str, user_query: str) -> str:
    if not report.strip():
        return ""
    prompt = (
        f"用户问题:{user_query}\n"
        f"回答内容:{report[:4000]}\n"
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
        report_mode=bool(state.get("report_mode")),
        markdown_path=markdown_path,
    )
    append_session_turn(record, session_id, runtime.settings)


async def reporter_node(state: AgentState, runtime: AgentRuntime) -> dict:
    log = node_logger(runtime.settings, "reporter")
    query = state.get("user_query", "")
    chunks = state.get("knowledge_chunks") or []
    meta = state.get("meta_hits") or []
    data_steps = state.get("data_tool_steps") or []
    plan_context = state.get("plan_context") or summarize_plan_steps(state.get("plan_steps") or [])
    process_result = state.get("process_result") or summarize_data_tool_steps(data_steps)
    processed = state.get("processed_data") or []
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
            log.info("检索已启用但未达到相似度阈值", min_score=runtime.settings.min_retrieval_score)
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

    if report_step < max_report_steps and not state.get("report_done"):
        prompt = build_reporter_decision_prompt(state, runtime)
        log.debug("调用 LLM 决定报告步骤", report_step=report_step)
        raw = await invoke_llm_decision(runtime.llm, prompt, phase="reporter")
        parsed = parse_reporter_decision_response(raw)
        action = parsed["action"]
        tool_name = parsed["tool_name"]
        params = parsed["params"]
        text_q = parsed["text_query"]
        data_q = parsed["data_query"]
        log.info("LLM 报告决策", action=action, tool_name=tool_name if action == "call_tool" else "")

        if action == "need_retrieval" and (text_q or data_q):
            log.info("触发补充检索", text_query=text_q, data_query=data_q)
            log.end(need_more_retrieval=True, next_node="retriever")
            return {
                "text_query": text_q or state.get("text_query", ""),
                "data_query": data_q or state.get("data_query", ""),
                "retrieval_round": retrieval_round + 1,
                "need_more_retrieval": True,
                "pending_tool": None,
                **append_node(state, "reporter"),
            }

        if action == "call_tool" and tool_name:
            if tool_name == "read_data_file" and not params.get("path"):
                catalog = get_session_catalog(state.get("session_id", ""), runtime.settings)
                if catalog:
                    params["path"] = next(iter(catalog))
            log.info("触发 report tool 调用", tool_name=tool_name, params=params)
            log.end(report_done=False, next_node="report_tool", tool_name=tool_name)
            return {
                "report_done": False,
                "pending_tool": PendingToolCall(phase="report", tool_name=tool_name, params=params),
                **append_node(state, "reporter"),
            }

    log.info("正在生成最终回答", report_mode=bool(state.get("report_mode")))
    answer_prompt = build_answer_prompt(state, runtime)
    answer, answer_summary = await invoke_llm_report_output(runtime.llm, answer_prompt)
    if not answer_summary:
        answer_summary = await _fallback_summary(runtime.llm, answer, state.get("user_query", ""))
    refs = list({c.chunk.source_file for c in chunks if c.chunk.source_file})
    refs += [m.record.file_name for m in meta]

    report_mode = state.get("report_mode") or (
        state.get("node_flags") and state.get("node_flags").enable_report
    )
    final_status = state.get("status", "ok")
    chart_paths = [c.path for c in charts if c.path]

    if report_mode:
        session_id = state.get("session_id", "default")
        report_dir = runtime.settings.report_output_path / session_id
        report_dir.mkdir(parents=True, exist_ok=True)
        md_path = report_dir / "report.md"
        chart_md = []
        for c in charts:
            if c.path:
                label = c.description or c.title or "chart"
                chart_md.append(f"![{label}]({c.path})")
        lines = [
            "# 分析报告",
            "## 分析",
            answer,
            "## 参考文档",
            *[f"- {r}" for r in refs],
        ]
        if chart_md:
            lines.extend(["## 图表", *chart_md])
        md_path.write_text("\n\n".join(lines), encoding="utf-8")
        log.info("Markdown 报告已写入", path=str(md_path))
        await _persist_turn(
            state,
            runtime,
            answer=answer,
            answer_summary=answer_summary,
            markdown_path=str(md_path),
        )
        artifact = ReportArtifact(
            session_id=session_id,
            markdown_path=str(md_path),
            image_paths=chart_paths,
            reference_docs=refs,
            nodes_traversed=state.get("nodes_traversed") or [],
        )
        log.end(report_done=True, report_mode=True, markdown_path=str(md_path))
        return {
            "final_answer": "详见报告",
            "report_artifact": artifact,
            "status": final_status,
            "need_more_retrieval": False,
            "report_done": True,
            "pending_tool": None,
            "report_context": {**summarize_report_steps(report_steps), "answer_summary": answer_summary},
            **append_node(state, "reporter"),
        }

    await _persist_turn(state, runtime, answer=answer, answer_summary=answer_summary)
    log.info("回答生成成功", answer_len=len(answer))
    log.end(report_done=True, status=final_status, answer_len=len(answer))
    return {
        "final_answer": answer,
        "status": final_status,
        "need_more_retrieval": False,
        "report_done": True,
        "pending_tool": None,
        "report_context": {"answer_summary": answer_summary},
        **append_node(state, "reporter"),
    }


async def debug_reporter_node() -> None:
    state = sample_state(plan_context={"step_count": 1, "latest": {"result": "2026-01-01"}})
    runtime = stub_runtime()
    result = await reporter_node(state, runtime)
    print_node_result("reporter_node", result)


if __name__ == "__main__":
    asyncio.run(debug_reporter_node())
