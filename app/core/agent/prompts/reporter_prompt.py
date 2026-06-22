"""Reporter LLM prompt builders."""

from __future__ import annotations

import json
from pathlib import Path

from app.core.agent.nodes._helpers import (
    extract_history_context,
    format_processed_data_full_for_prompt,
    format_report_tool_history,
    normalize_queries_for_flags,
    report_tool_catalog,
    summarize_data_tool_steps,
    summarize_plan_steps,
    user_require_text,
)
from app.core.agent.state import AgentRuntime, AgentState, dedupe_scored_chunks, dedupe_scored_meta
from app.core.session.process_artifact_store import (
    format_intermediate_catalog_for_agent,
    get_session_catalog,
    resolve_catalog_path,
)


def _format_document_chunks(chunks: list, *, limit: int = 8, content_len: int = 400) -> str:
    unique = dedupe_scored_chunks(chunks)
    if not unique:
        return "（暂无文档片段）"
    lines: list[str] = []
    for i, c in enumerate(unique[:limit], 1):
        src = c.chunk.source_file or c.chunk.doc_id or "unknown"
        lines.append(f"{i}. [{src}] {c.chunk.content[:content_len]}")
    if len(unique) > limit:
        lines.append(f"... 共 {len(unique)} 条，仅展示前 {limit} 条")
    return "\n".join(lines)


def _format_meta_hits(meta: list, *, limit: int = 8) -> str:
    unique = dedupe_scored_meta(meta)
    if not unique:
        return "（暂无结构化元数据命中）"
    lines: list[str] = []
    for i, m in enumerate(unique[:limit], 1):
        name = m.record.file_name or m.record.file_path
        preview = (m.record.search_text or "")[:200]
        lines.append(f"{i}. [{name}] {preview}")
    if len(unique) > limit:
        lines.append(f"... 共 {len(unique)} 条，仅展示前 {limit} 条")
    return "\n".join(lines)


def _format_supplemental_loaded_data(
    report_context: dict,
    *,
    already_loaded: set[str],
    max_per_file: int,
) -> str:
    loaded_files = report_context.get("loaded_files") or {}
    if not loaded_files:
        loaded = str(report_context.get("loaded_content", "") or "")
        path = str(report_context.get("loaded_path", "") or "")
        if loaded and path and path not in already_loaded:
            name = Path(path).name
            return f"#### {name}\n{loaded[:max_per_file]}"
        return ""
    blocks: list[str] = []
    for path, content in loaded_files.items():
        if path in already_loaded:
            continue
        name = Path(str(path)).name
        blocks.append(f"#### {name}\n{str(content)[:max_per_file]}")
    return "\n\n".join(blocks)
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
    chunks = dedupe_scored_chunks(state.get("knowledge_chunks") or [])
    sources = sorted({c.chunk.source_file for c in chunks if c.chunk.source_file})
    if sources:
        lines.append(f"- 已命中文档来源: {', '.join(sources[:30])}")
    meta = dedupe_scored_meta(state.get("meta_hits") or [])
    files = sorted({m.record.file_name or m.record.file_path for m in meta if m.record.file_name or m.record.file_path})
    if files:
        lines.append(f"- 已命中结构化文件: {', '.join(files[:30])}")
    return "\n".join(lines) if lines else "（尚无检索记录）"


def build_reporter_prompt(
    state: AgentState,
    runtime: AgentRuntime,
    *,
    force_done: bool = False,
    skip_read_data_file: bool = False,
) -> str:
    chunks = dedupe_scored_chunks(state.get("knowledge_chunks") or [])
    meta = dedupe_scored_meta(state.get("meta_hits") or [])
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
    max_chars = runtime.settings.context_size_threshold_chars
    processed_full_text, loaded_paths = format_processed_data_full_for_prompt(state, runtime.settings)
    supplemental_data = _format_supplemental_loaded_data(
        report_context,
        already_loaded=set(loaded_paths),
        max_per_file=max_chars,
    )
    process_summary = json.dumps(process_result, ensure_ascii=False)[:1500] if process_result else "（无）"

    force_line = ""
    if force_done:
        force_line = (
            f"已达最大 report tool 步数({max_report_steps})，必须 action=done 并输出最终 answer 与 summary，"
            "禁止 call_tool、retrieve_text、retrieve_data。\n"
        )
    skip_read_line = ""
    if skip_read_data_file and not force_done:
        skip_read_line = (
            "「已处理数据全量」已注入下方，禁止对 catalog 内文件再 call_tool read_data_file；"
            "若仍需公司/标的文本资料，须 action=retrieve_text。\n"
        )

    supplemental_block = ""
    if supplemental_data.strip():
        supplemental_block = f"补充读取的数据:\n{supplemental_data}\n"

    return (
        f"用户需求:{user_require_text(state)}\n"
        f"规划结果:{json.dumps(plan_context, ensure_ascii=False)[:800]}\n"

        f"中间数据（文件名:描述）:\n{catalog_text}\n"
        f"已处理数据全量（单文件/总量上限 {max_chars} 字符，超出截断）:\n{processed_full_text}\n"
        f"{supplemental_block}"
        f"处理摘要:\n{process_summary}\n"
        f"文档片段:\n{doc_text}\n"
        f"可用 report 工具:\n{tool_catalog}\n"
        f"已检索记录（补充检索时须避免重复 query 与已命中来源）:\n{retrieval_history}\n"
        f"{force_line}"
        f"{skip_read_line}"
        "请根据问题需求，判断是否已有可用数据/文档片段。按以下规则填写 JSON 各字段。\n"
        "规则：\n"
        "0. 二次文本检索（重要）：若用户需求含「相关资料/背景/研报/公告/新闻/解读/资料」等，"
        "且「已处理数据全量」或处理摘要中已识别出具体公司或证券名称，但「文档片段」几乎没有该公司相关内容："
        "必须先 action=retrieve_text，text_query 填「{公司名} 研报/公告/资料」等（勿重复已检索 query），"
        "answer/summary 留空；禁止直接 done。"
        "若首轮规划未做知识检索（文档片段为空或仅有榜单/指标泛化内容），而回答需要目标公司文本资料，同样必须先 retrieve_text。\n"
        "1. 若仍需补充结构化数据检索，action=retrieve_data 并填写 data_query（须与已检索记录不同）；"
        "同时填写 enable_process、enable_chart 表示补检索后是否需重新数据处理/画图；"
        "不得填写 text_query。\n"
        "2. 若仍需补充知识文本检索，action=retrieve_text 并填写 text_query（须避免与已检索记录重复）；"
        "不得填写 data_query、enable_process、enable_chart。\n"
        "3. 一般禁止 read_data_file：已处理数据全量已在上方注入。"
        "仅当需读取 catalog 未包含的额外文件时才 action=call_tool read_data_file。\n"
        "4. 信息已足够时 action=done，并填写 answer 与 summary。"
        "「足够」指：已处理数据全量与文档片段已满足问题，或无需更多文本资料。"
        "若「已检索记录」已有补充检索且文档片段非空，或补充检索连续无新结果：禁止再 retrieve，必须 done 并基于已有内容作答。\n"
        "5. action 为 retrieve_text、retrieve_data 或 call_tool 时，answer、summary 必须均为空字符串。\n"
        "6. action 为 done 时：根据用户需求、已处理数据全量与文档片段给出专业金融分析建议，并作适当拓展，填入 answer；summary 填 2-3 句摘要。\n"
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
