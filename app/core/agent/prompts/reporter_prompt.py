"""Reporter LLM prompt builders."""

from __future__ import annotations

import json

from app.core.agent.nodes._helpers import (
    extract_history_context,
    format_report_tool_history,
    report_tool_catalog,
    summarize_data_tool_steps,
    summarize_plan_steps,
)
from app.core.agent.state import AgentRuntime, AgentState
from app.core.session.process_artifact_store import format_intermediate_catalog, get_session_catalog


def build_reporter_decision_prompt(state: AgentState, runtime: AgentRuntime) -> str:
    chunks = state.get("knowledge_chunks") or []
    meta = state.get("meta_hits") or []
    data_steps = state.get("data_tool_steps") or []
    plan_context = state.get("plan_context") or summarize_plan_steps(state.get("plan_steps") or [])
    process_result = state.get("process_result") or summarize_data_tool_steps(data_steps)
    report_steps = state.get("report_steps") or []
    report_step = state.get("report_step", 0)
    max_report_steps = state.get("max_report_tool_steps", runtime.settings.max_report_tool_steps)
    report_context = dict(state.get("report_context") or {})
    charts = state.get("chart_artifacts") or []
    data_desc = state.get("dataprocessplan", "")
    history = format_report_tool_history(report_steps)
    tool_catalog = report_tool_catalog(runtime)
    catalog_text = format_intermediate_catalog(
        get_session_catalog(state.get("session_id", ""), runtime.settings)
    )
    chart_paths = [c.path for c in charts if c.path]

    return (
        f"问题:{state.get('user_query','')}\n"
        f"规划结果:{json.dumps(plan_context, ensure_ascii=False)[:800]}\n"
        f"dataprocessplan:{data_desc}\n"
        f"处理流程摘要:{json.dumps(process_result, ensure_ascii=False)[:800]}\n"
        f"中间数据（路径:描述）:\n{catalog_text}\n"
        f"图表路径:{chart_paths}\n"
        f"文档片段数:{len(chunks)} 元数据命中:{len(meta)} 图表:{len(charts)}\n"
        f"已加载报告上下文:{json.dumps(report_context, ensure_ascii=False)[:600]}\n"
        f"已执行报告工具({report_step}/{max_report_steps}):\n{history}\n"
        "输出 JSON："
        '{"action":"call_tool|done|need_retrieval","tool_name":"read_data_file",'
        '"params":{},"text_query":"","data_query":""}\n'
        f"可用 report 工具:\n{tool_catalog}\n"
        "需要读取中间数据全文时用 read_data_file（params.path 为绝对路径）；"
        "需要补充检索时 action=need_retrieval 并填写 text_query/data_query；"
        "信息足够生成报告时 action=done。"
    )


def build_answer_prompt(state: AgentState, runtime: AgentRuntime) -> str:
    chunks = state.get("knowledge_chunks") or []
    meta = state.get("meta_hits") or []
    plan_context = state.get("plan_context") or summarize_plan_steps(state.get("plan_steps") or [])
    history_ctx = extract_history_context(state.get("plan_steps") or [])
    history_block = history_ctx.get("context_text", "") if history_ctx else ""
    report_context = dict(state.get("report_context") or {})
    charts = state.get("chart_artifacts") or []
    chart_lines = [f"- {c.path}" for c in charts if c.path]
    catalog_text = format_intermediate_catalog(
        get_session_catalog(state.get("session_id", ""), runtime.settings)
    )
    context_parts = [c.chunk.content[:400] for c in chunks[:5]]
    meta_parts = [f"{m.record.file_name}: {m.record.search_text[:200]}" for m in meta[:5]]
    loaded = report_context.get("loaded_content", "")
    report_mode = bool(
        state.get("report_mode") or (state.get("node_flags") and state.get("node_flags").enable_report)
    )

    body_hint = "Markdown 报告正文" if report_mode else "回答正文"
    return (
        f"问题:{state.get('user_query','')}\n"
        f"历史上下文:\n{history_block}\n"
        f"规划结果:{plan_context}\n"
        f"文档片段:{context_parts}\n元数据:{meta_parts}\n"
        f"中间数据（路径:描述）:\n{catalog_text}\n"
        f"报告工具加载内容:{loaded}\n"
        f"图表路径:\n" + "\n".join(chart_lines) + "\n"
        f"请基于上述信息生成{body_hint}，并同时给出 2-3 句中文摘要。\n"
        "输出 JSON（不要其它文字）："
        '{"report":"' + body_hint + '（Markdown，可在合适位置用 ![描述](图表路径) 插入图表）",'
        '"summary":"2-3句中文摘要，含核心结论与关键数据"}'
    )


def parse_reporter_decision_response(raw: str) -> dict:
    from app.core.agent.nodes._helpers import parse_llm_json

    action = "done"
    tool_name = "read_data_file"
    params: dict = {}
    text_q = ""
    data_q = ""
    try:
        data = parse_llm_json(raw)
        action = data.get("action", "done")
        tool_name = data.get("tool_name", tool_name)
        params = dict(data.get("params") or {})
        text_q = str(data.get("text_query", "") or "")
        data_q = str(data.get("data_query", "") or "")
    except Exception:
        action = "done"
    return {
        "action": action,
        "tool_name": tool_name,
        "params": params,
        "text_query": text_q,
        "data_query": data_q,
    }
