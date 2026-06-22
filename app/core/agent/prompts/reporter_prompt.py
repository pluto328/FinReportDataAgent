"""Reporter LLM prompt builders."""

from __future__ import annotations

import json
from pathlib import Path

from app.core.agent.nodes._helpers import (
    extract_history_context,
    format_file_previews_for_prompt,
    format_report_tool_history,
    normalize_queries_for_flags,
    report_tool_catalog,
    summarize_data_tool_steps,
    summarize_plan_steps,
    user_require_text,
)
from app.schemas.structured import REPORTER_PREVIEW_DISPLAY_ROWS
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


def _format_loaded_data(report_context: dict, *, max_per_file: int = 3500) -> str:
    loaded_files = report_context.get("loaded_files") or {}
    if loaded_files:
        blocks: list[str] = []
        for path, content in loaded_files.items():
            name = Path(str(path)).name
            text = str(content)[:max_per_file]
            blocks.append(f"#### {name}\n{text}")
        return "\n\n".join(blocks)
    loaded = str(report_context.get("loaded_content", "") or "")
    if loaded:
        return loaded[:max_per_file]
    return "（尚未通过 read_data_file 读取中间数据）"


def _format_loaded_paths(report_context: dict) -> str:
    paths = report_context.get("loaded_paths") or []
    if not paths:
        single = report_context.get("loaded_path", "")
        if single:
            paths = [single]
    if not paths:
        return "（无）"
    return ", ".join(Path(str(p)).name for p in paths)


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
    data_steps = state.get("data_tool_steps") or []
    process_result = state.get("process_result") or summarize_data_tool_steps(data_steps)
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
    loaded_paths_text = _format_loaded_paths(report_context)
    loaded_data_text = _format_loaded_data(report_context)
    previews_text = format_file_previews_for_prompt(
        state.get("file_previews"),
        display_rows=REPORTER_PREVIEW_DISPLAY_ROWS,
    )
    process_summary = json.dumps(process_result, ensure_ascii=False)[:1500] if process_result else "（无）"

    force_line = ""
    if force_done:
        force_line = (
            f"已达最大 report tool 步数({max_report_steps})，必须 action=done 并输出最终 answer 与 summary，"
            "禁止 call_tool、retrieve_text、retrieve_data。\n"
        )

    return (
        f"用户需求:{user_require_text(state)}\n"
        f"规划结果:{json.dumps(plan_context, ensure_ascii=False)[:800]}\n"

        f"中间数据（文件名:描述）:\n{catalog_text}\n"
        f"已处理数据预览(CSV):\n{previews_text}\n"
        f"处理摘要:\n{process_summary}\n"
        f"已读取文件:{loaded_paths_text}\n"
        f"已读取数据内容:\n{loaded_data_text}\n"
        f"文档片段:\n{doc_text}\n"
        f"可用 report 工具:\n{tool_catalog}\n"
        f"已检索记录（补充检索时须避免重复 query 与已命中来源）:\n{retrieval_history}\n"
        f"{force_line}"
        "请根据问题需求，判断是否已有可用数据。按以下规则填写 JSON 各字段。\n"
        "规则：\n"
        "1. 若仍需补充结构化数据检索，action=retrieve_data 并填写 data_query（须与已检索记录不同）；"
        "同时填写 enable_process、enable_chart 表示补检索后是否需重新数据处理/画图；"
        "不得填写 text_query。\n"
        "2. 若仍需补充知识文本检索，action=retrieve_text 并填写 text_query（须避免与已检索记录重复）；"
        "不得填写 data_query、enable_process、enable_chart。\n"
        "3. 需求需要读取中间文件回答时，action=call_tool，tool_name=read_data_file，params.path 填中间数据文件名；"
        "若「已处理数据预览(CSV)」已含所需数据，禁止 read_data_file，直接 action=done。\n"
        "4. 信息已足够时 action=done，并填写 answer 与 summary。\n"
        "5. action 为 retrieve_text、retrieve_data 或 call_tool 时，answer、summary 必须均为空字符串。\n"
        "6. action 为 done 时：根据用户需求、已处理数据给出专业金融分析建议，并作适当拓展，填入 answer；summary 填 2-3 句摘要。\n"
        "输出 JSON（不要其它文字）："
        '{"action":"call_tool|retrieve_text|retrieve_data|done","tool_name":"read_data_file",'
        '"params":{},"text_query":"","data_query":"","enable_process":false,"enable_chart":false,'
        '"answer":"","summary":""}'
    )


def parse_reporter_response(raw: str) -> dict:
    from app.core.agent.nodes._helpers import parse_llm_json
    from app.schemas.structured import NodeEnableFlags

    action = "done"
    tool_name = "read_data_file"
    params: dict = {}
    text_q = ""
    data_q = ""
    enable_process = False
    enable_chart = False
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
        enable_process = bool(data.get("enable_process", False))
        enable_chart = bool(data.get("enable_chart", False))
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
        summary = ""

    if action == "retrieve_text":
        data_q = ""
        enable_process = False
        enable_chart = False
        retrieve_flags = NodeEnableFlags(enable_knowledge_retrieve=True)
        text_q, data_q = normalize_queries_for_flags(text_q, data_q, retrieve_flags)
        if not text_q:
            action = "done"
    elif action == "retrieve_data":
        text_q = ""
        retrieve_flags = NodeEnableFlags(enable_data_retrieve=True)
        text_q, data_q = normalize_queries_for_flags(text_q, data_q, retrieve_flags)
        if not data_q:
            action = "done"

    return {
        "action": action,
        "tool_name": tool_name,
        "params": params,
        "text_query": text_q,
        "data_query": data_q,
        "enable_process": enable_process,
        "enable_chart": enable_chart,
        "answer": answer,
        "report": report,
        "summary": summary,
    }
