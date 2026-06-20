"""Reporter LLM prompt builders."""

from __future__ import annotations

import json

from app.core.agent.nodes._helpers import (
    extract_history_context,
    format_report_tool_history,
    report_tool_catalog,
    summarize_plan_steps,
)
from app.core.agent.state import AgentRuntime, AgentState
from app.core.session.process_artifact_store import (
    format_intermediate_catalog_for_agent,
    get_session_catalog,
    resolve_catalog_path,
)


def _format_document_chunks(chunks: list, *, limit: int = 8, content_len: int = 400) -> str:
    if not chunks:
        return "（暂无文档片段）"
    lines: list[str] = []
    for i, c in enumerate(chunks[:limit], 1):
        src = c.chunk.source_file or c.chunk.doc_id or "unknown"
        lines.append(f"{i}. [{src}] {c.chunk.content[:content_len]}")
    if len(chunks) > limit:
        lines.append(f"... 共 {len(chunks)} 条，仅展示前 {limit} 条")
    return "\n".join(lines)


def _format_meta_hits(meta: list, *, limit: int = 8) -> str:
    if not meta:
        return "（暂无结构化元数据命中）"
    lines: list[str] = []
    for i, m in enumerate(meta[:limit], 1):
        name = m.record.file_name or m.record.file_path
        preview = (m.record.search_text or "")[:200]
        lines.append(f"{i}. [{name}] {preview}")
    if len(meta) > limit:
        lines.append(f"... 共 {len(meta)} 条，仅展示前 {limit} 条")
    return "\n".join(lines)


def _format_retrieval_history(state: AgentState) -> str:
    lines: list[str] = []
    text_q = state.get("text_query", "")
    data_q = state.get("data_query", "")
    if text_q:
        lines.append(f"- 规划/当前文本检索 query: {text_q}")
    if data_q:
        lines.append(f"- 规划/当前数据检索 query: {data_q}")
    ctx = state.get("report_context") or {}
    for q in ctx.get("supplemental_text_queries") or []:
        lines.append(f"- 已补充文本检索: {q}")
    for q in ctx.get("supplemental_data_queries") or []:
        lines.append(f"- 已补充数据检索: {q}")
    chunks = state.get("knowledge_chunks") or []
    sources = sorted({c.chunk.source_file for c in chunks if c.chunk.source_file})
    if sources:
        lines.append(f"- 已命中文档来源: {', '.join(sources[:30])}")
    meta = state.get("meta_hits") or []
    files = sorted({m.record.file_name or m.record.file_path for m in meta if m.record.file_name or m.record.file_path})
    if files:
        lines.append(f"- 已命中结构化文件: {', '.join(files[:30])}")
    return "\n".join(lines) if lines else "（尚无检索记录）"


def build_reporter_prompt(
    state: AgentState,
    runtime: AgentRuntime,
    *,
    force_done: bool = False,
) -> str:
    chunks = state.get("knowledge_chunks") or []
    meta = state.get("meta_hits") or []
    report_steps = state.get("report_steps") or []
    report_step = state.get("report_step", 0)
    max_report_steps = state.get("max_report_tool_steps", runtime.settings.max_report_tool_steps)
    report_context = dict(state.get("report_context") or {})
    charts = state.get("chart_artifacts") or []
    data_desc = state.get("data_process_plan", "")
    plan_context = state.get("plan_context") or summarize_plan_steps(state.get("plan_steps") or [])
    history_ctx = extract_history_context(state.get("plan_steps") or [])
    history_block = history_ctx.get("context_text", "") if history_ctx else ""
    history = format_report_tool_history(report_steps)
    tool_catalog = report_tool_catalog(runtime)
    catalog_text = format_intermediate_catalog_for_agent(
        get_session_catalog(state.get("session_id", ""), runtime.settings)
    )
    chart_paths = [c.path for c in charts if c.path]
    doc_text = _format_document_chunks(chunks)
    meta_text = _format_meta_hits(meta)
    retrieval_history = _format_retrieval_history(state)
    loaded = report_context.get("loaded_content", "")
    report_mode = bool(
        state.get("report_mode") or (state.get("node_flags") and state.get("node_flags").enable_report)
    )

    force_line = ""
    if force_done:
        force_line = f"已达最大工具步数({max_report_steps})，必须 action=done 并输出最终 answer/report/summary。\n"

    report_rule = (
        'report_mode=true：answer 固定填 "详见报告"，report 填 Markdown 报告正文（可插入 ![描述](图表路径)），summary 填 2-3 句摘要。'
        if report_mode
        else "report_mode=false：answer 填回答正文，report 留空，summary 填 2-3 句摘要。"
    )

    return (
        f"问题:{state.get('user_query', '')}\n"
        f"report_mode:{report_mode}\n"
        f"历史上下文:\n{history_block}\n"
        f"规划结果:{json.dumps(plan_context, ensure_ascii=False)[:800]}\n"
        f"data_process_plan:{data_desc}\n"
        f"中间数据（文件名:描述）:\n{catalog_text}\n"
        f"文档片段:\n{doc_text}\n"
        f"图表路径:{chart_paths}\n"
        f"工具调用历史({report_step}/{max_report_steps}):\n{history}\n"
        f"可用 report 工具:\n{tool_catalog}\n"
        f"已检索记录（补充检索时须避免重复 query 与已命中来源）:\n{retrieval_history}\n"
        f"{force_line}"
        "请根据问题、data_process_plan 与上述上下文决定下一步，并按要求填写 JSON 各字段。\n"
        "规则：\n"
        "1. 若仍需补充结构化数据检索，action=retrieve_data 并填写 data_query（可与 text_query 无关，单独检索）；"
        "query 须与已检索记录中的 query 不同，且应换角度/换关键词以获取未命中的数据。\n"
        "2. 若仍需补充知识文本检索，action=retrieve_text 并填写 text_query（可与 data_query 无关，单独检索）；"
        "query 须避免与已检索记录重复，并尽量覆盖尚未命中的文档来源。\n"
        "3. 若需读取中间数据全文，action=call_tool，tool_name=read_data_file，"
        "params.path 填上方中间数据列表中的文件名（不要猜路径、不要自行加 _processed 后缀）。\n"
        "4. 信息已足够时 action=done，并填写 answer/report/summary。\n"
        "5. action 为 retrieve_text、retrieve_data 或 call_tool 时，answer、report、summary 必须均为空字符串。\n"
        f"6. action 为 done 时：{report_rule}\n"
        "输出 JSON（不要其它文字）："
        '{"action":"call_tool|retrieve_text|retrieve_data|done","tool_name":"read_data_file",'
        '"params":{},"text_query":"","data_query":"","answer":"","report":"","summary":""}'
    )


def parse_reporter_response(raw: str) -> dict:
    from app.core.agent.nodes._helpers import parse_llm_json

    action = "done"
    tool_name = "read_data_file"
    params: dict = {}
    text_q = ""
    data_q = ""
    answer = ""
    report = ""
    summary = ""
    try:
        data = parse_llm_json(raw)
        action = str(data.get("action", "done") or "done")
        tool_name = data.get("tool_name", tool_name)
        params = dict(data.get("params") or {})
        text_q = str(data.get("text_query", "") or "").strip()
        data_q = str(data.get("data_query", "") or "").strip()
        answer = str(data.get("answer", "") or "").strip()
        report = str(data.get("report", "") or "").strip()
        summary = str(data.get("summary", "") or "").strip()
    except Exception:
        action = "done"

    if action == "need_retrieval":
        if text_q and not data_q:
            action = "retrieve_text"
        elif data_q and not text_q:
            action = "retrieve_data"
        elif text_q or data_q:
            action = "retrieve_text" if text_q else "retrieve_data"

    if action in ("retrieve_text", "retrieve_data", "call_tool"):
        answer = ""
        report = ""
        summary = ""

    if action == "retrieve_text" and not text_q:
        action = "done"
    if action == "retrieve_data" and not data_q:
        action = "done"

    return {
        "action": action,
        "tool_name": tool_name,
        "params": params,
        "text_query": text_q,
        "data_query": data_q,
        "answer": answer,
        "report": report,
        "summary": summary,
    }
