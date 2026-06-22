"""Shared helpers for LangGraph agent nodes."""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from app.core.agent.state import AgentRuntime, AgentState
from app.schemas.structured import (
    DATA_PROCESSOR_PREVIEW_DISPLAY_ROWS,
    FILE_PREVIEW_STORE_ROWS,
    REPORTER_PREVIEW_DISPLAY_ROWS,
    DataToolStepResult,
    FilePreviewEntry,
    NodeEnableFlags,
    PlanStepResult,
    ReportStepResult,
)

if TYPE_CHECKING:
    from app.config.settings import Settings

_SINGLE_PATH_TOOLS = frozenset({"data_filter", "make_chart"})
_MULTI_PATH_TOOLS = frozenset({"sql_execute", "pandas_execute", "preview_read"})


def user_require_text(state: AgentState) -> str:
    return str(state.get("user_require") or state.get("user_query") or "").strip()


def parse_user_require(data: dict[str, Any], *, fallback: str = "") -> str:
    raw = data.get("user_require")
    if isinstance(raw, dict):
        parts: list[str] = []
        for key, label in (
            ("data", "需要什么数据"),
            ("knowledge", "需要什么知识"),
            ("tables", "需要什么表格"),
        ):
            val = str(raw.get(key, "") or "").strip()
            if val:
                parts.append(f"{label}：{val}")
        text = "；".join(parts)
        return text or fallback
    text = str(raw or "").strip()
    return text or fallback


def normalize_queries_for_flags(
    text_q: str,
    data_q: str,
    flags: NodeEnableFlags,
    *,
    fallback: str = "",
) -> tuple[str, str]:
    text = text_q.strip() if flags.enable_knowledge_retrieve else ""
    data = data_q.strip() if flags.enable_data_retrieve else ""
    if flags.enable_knowledge_retrieve and not text and fallback:
        text = fallback.strip()
    if flags.enable_data_retrieve and not data and fallback:
        data = fallback.strip()
    return text, data


def append_node(state: AgentState, name: str) -> dict:
    return {"nodes_traversed": [name]}


def parse_llm_json(raw: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", raw, re.S)
    return json.loads(match.group(0) if match else raw)


def normalize_data_tool_params(
    params: dict,
    *,
    file_paths: list[str],
    tool_name: str = "",
    session_id: str = "",
    settings: Settings | None = None,
    extra_paths: list[str] | None = None,
) -> dict:
    from app.core.session.process_artifact_store import resolve_catalog_path

    out = dict(params or {})
    retrieved = [str(p) for p in file_paths if p]
    search_pool = list(retrieved)
    if extra_paths:
        search_pool.extend(str(p) for p in extra_paths if p)

    def resolve_one(raw: str) -> str:
        text = str(raw or "").strip()
        if not text:
            return ""
        return resolve_catalog_path(session_id, text, settings, extra_paths=search_pool)

    if tool_name == "preview_read":
        raw_paths: list[str] = []
        tool_paths = out.get("file_paths")
        single = out.get("file_path")
        if tool_paths:
            if isinstance(tool_paths, str):
                raw_paths = [tool_paths]
            else:
                raw_paths = [str(p) for p in tool_paths if p]
        elif single:
            if isinstance(single, list):
                raw_paths = [str(p) for p in single if p]
            else:
                raw_paths = [str(single)]
        resolved_paths: list[str] = []
        for raw in raw_paths:
            hit = resolve_one(raw)
            if hit and hit not in resolved_paths:
                resolved_paths.append(hit)
        result = dict(out)
        result["file_paths"] = resolved_paths
        if resolved_paths:
            result["file_path"] = resolved_paths[0]
        elif single and not isinstance(single, list):
            result["file_path"] = str(single)
        return result

    raw_paths: list[str] = []
    tool_paths = out.get("file_paths")
    single = out.get("file_path")
    if tool_paths:
        if isinstance(tool_paths, str):
            raw_paths = [tool_paths]
        else:
            raw_paths = [str(p) for p in tool_paths if p]
    elif single:
        if isinstance(single, list):
            raw_paths = [str(p) for p in single if p]
        else:
            raw_paths = [str(single)]
    elif tool_name in _MULTI_PATH_TOOLS and retrieved:
        raw_paths = list(retrieved)

    resolved_paths: list[str] = []
    for raw in raw_paths:
        hit = resolve_one(raw)
        if hit and hit not in resolved_paths:
            resolved_paths.append(hit)

    if tool_name in _SINGLE_PATH_TOOLS:
        raw_single = str(single or (raw_paths[0] if raw_paths else "") or "").strip()
        resolved = resolve_one(raw_single) if raw_single else ""
        out["file_path"] = resolved or raw_single
        out.pop("file_paths", None)
        return out

    out["file_paths"] = resolved_paths
    if resolved_paths:
        out["file_path"] = resolved_paths[0]
    elif single and not isinstance(single, list):
        out["file_path"] = str(single)
    return out


def summarize_data_tool_steps(steps: list[DataToolStepResult]) -> dict[str, Any]:
    if not steps:
        return {}
    return {
        "step_count": len(steps),
        "steps": [
            {
                "step": s.step,
                "tool": s.tool_name,
                "result": s.result,
                "error": s.error,
            }
            for s in steps
        ],
        "latest": steps[-1].result,
    }


def summarize_plan_steps(steps: list[PlanStepResult]) -> dict[str, Any]:
    if not steps:
        return {}
    base = {
        "step_count": len(steps),
        "steps": [
            {
                "step": s.step,
                "method": s.method,
                "result": s.result,
                "error": s.error,
            }
            for s in steps
        ],
        "latest": steps[-1].result,
    }
    history_ctx = extract_history_context(steps)
    if history_ctx:
        base["history_context"] = history_ctx
    return base


def extract_history_context(steps: list[PlanStepResult]) -> dict[str, Any]:
    for s in reversed(steps):
        if s.method != "load_history_context" or s.error:
            continue
        result = s.result or {}
        if result.get("context_text") or result.get("history"):
            return {
                "context_text": result.get("context_text", ""),
                "history": result.get("history", []),
                "turn_count": result.get("turn_count", 0),
            }
    return {}


def summarize_report_steps(steps: list[ReportStepResult]) -> dict[str, Any]:
    if not steps:
        return {}
    return {
        "step_count": len(steps),
        "steps": [
            {
                "step": s.step,
                "tool": s.tool_name,
                "result": s.result,
                "error": s.error,
            }
            for s in steps
        ],
        "latest": steps[-1].result,
    }



def _sanitize_preview_rows(rows: list[Any]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for item in rows[:FILE_PREVIEW_STORE_ROWS]:
        if isinstance(item, dict):
            cleaned.append(item)
    return cleaned


def load_preview_rows_from_path(path: str, *, rows: int = FILE_PREVIEW_STORE_ROWS) -> list[dict[str, Any]]:
    from app.core.tools.structured_ops import read_table_preview

    text = str(path or "").strip()
    if not text:
        return []
    try:
        df = read_table_preview(text, rows=rows)
        return _sanitize_preview_rows(df.to_dict(orient="records"))
    except Exception:
        return []


def strip_preview_from_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    out = dict(result or {})
    out.pop("preview", None)
    return out


def normalize_file_previews(
    previews: dict[str, FilePreviewEntry | dict[str, Any]] | None,
) -> dict[str, FilePreviewEntry]:
    if not previews:
        return {}
    out: dict[str, FilePreviewEntry] = {}
    for name, entry in previews.items():
        if isinstance(entry, FilePreviewEntry):
            out[str(name)] = entry
        elif isinstance(entry, dict):
            out[str(name)] = FilePreviewEntry.model_validate(entry)
    return out


def upsert_file_preview(
    previews: dict[str, FilePreviewEntry | dict[str, Any]] | None,
    *,
    file_name: str,
    path: str,
    description: str,
    preview_rows: list[dict[str, Any]],
) -> dict[str, FilePreviewEntry]:
    merged = normalize_file_previews(previews)
    name = str(file_name or Path(path).name).strip()
    if not name:
        return merged
    merged[name] = FilePreviewEntry(
        path=str(path),
        description=str(description or ""),
        preview_rows=_sanitize_preview_rows(preview_rows),
        row_count=len(preview_rows),
    )
    return merged


def _preview_rows_to_csv(rows: list[dict[str, Any]], *, display_rows: int) -> str:
    sliced = [r for r in rows[:display_rows] if isinstance(r, dict)]
    if not sliced:
        return "（无有效列）"
    df = pd.DataFrame.from_records(sliced)
    if df.empty:
        return "（无有效列）"
    df = df.replace("", pd.NA).dropna(axis=1, how="all")
    if df.empty:
        return "（无有效列）"
    for col in df.columns:
        non_null = df[col].notna()
        if not non_null.any():
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric[non_null].notna().all():
            df[col] = numeric.round(0).map(lambda x: "" if pd.isna(x) else str(int(x)))
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().rstrip()


def format_file_previews_for_prompt(
    previews: dict[str, FilePreviewEntry] | dict[str, dict[str, Any]] | None,
    *,
    display_rows: int = DATA_PROCESSOR_PREVIEW_DISPLAY_ROWS,
) -> str:
    normalized = normalize_file_previews(previews)
    if not normalized:
        return "（暂无）"
    lines: list[str] = []
    for name, entry in normalized.items():
        preview_text = _preview_rows_to_csv(entry.preview_rows, display_rows=display_rows)
        desc = entry.description or "无描述"
        lines.append(f"- {name}|{desc}:\n{preview_text}")
    return "\n".join(lines)


def format_data_tool_history(steps: list[DataToolStepResult]) -> str:
    if not steps:
        return "（尚无数据处理工具执行记录）"
    lines: list[str] = []
    for s in steps:
        if s.error:
            lines.append(f"步骤{s.step} {s.tool_name} -> 失败: {s.error}")
            continue
        desc = str(s.params.get("artifact_description") or s.params.get("description") or "").strip()
        out_name = ""
        res = s.result or {}
        if res.get("paths"):
            out_name = ", ".join(Path(str(p)).name for p in res["paths"] if p)
        elif res.get("path"):
            out_name = Path(str(res["path"])).name
        elif s.tool_name == "preview_read":
            fps = s.params.get("file_paths") or []
            if isinstance(fps, list) and fps:
                out_name = ", ".join(Path(str(p)).name for p in fps if p)
            elif s.params.get("file_path"):
                out_name = Path(str(s.params.get("file_path") or "")).name
            desc = desc or "原始数据预览"
        lines.append(
            f"步骤{s.step} {s.tool_name} -> 文件: {out_name or '无'}; 描述: {desc or '无'}"
        )
    return "\n".join(lines)


def format_plan_history(steps: list[PlanStepResult]) -> str:
    if not steps:
        return "（尚无规划工具执行记录）"
    lines: list[str] = []
    for s in steps:
        payload = s.result if not s.error else {"error": s.error}
        lines.append(
            f"步骤{s.step} {s.method}({json.dumps(s.params, ensure_ascii=False)}) "
            f"-> {json.dumps(payload, ensure_ascii=False)[:800]}"
        )
    return "\n".join(lines)


def format_report_tool_history(steps: list[ReportStepResult]) -> str:
    if not steps:
        return "（尚无报告工具执行记录）"
    lines: list[str] = []
    for s in steps:
        if s.error:
            lines.append(
                f"步骤{s.step} {s.tool_name}({json.dumps(s.params, ensure_ascii=False)}) "
                f"-> 失败: {s.error}"
            )
            continue
        payload = s.result if isinstance(s.result, dict) else {}
        if s.tool_name == "read_data_file" and payload.get("content") is not None:
            path = payload.get("path", "")
            name = Path(path).name if path else "unknown"
            chars = payload.get("char_count", len(str(payload.get("content", ""))))
            truncated = "，已截断" if payload.get("truncated") else ""
            lines.append(
                f"步骤{s.step} read_data_file({name}) -> 成功，{chars} 字符{truncated}"
            )
        else:
            lines.append(
                f"步骤{s.step} {s.tool_name}({json.dumps(s.params, ensure_ascii=False)}) "
                f"-> {json.dumps(payload, ensure_ascii=False)[:400]}"
            )
    return "\n".join(lines)


def plan_tool_catalog(runtime: AgentRuntime) -> str:
    lines: list[str] = []
    for name in runtime.plan_registry.list_names():
        tool = runtime.plan_registry.get(name)
        if tool:
            lines.append(f"- {tool.name}: {tool.description}")
    return "\n".join(lines) if lines else "（无）"


def data_tool_catalog(runtime: AgentRuntime) -> str:
    lines: list[str] = []
    for name in runtime.data_registry.list_names():
        tool = runtime.data_registry.get(name)
        if tool:
            lines.append(f"- {tool.name}: {tool.description}")
    return "\n".join(lines) if lines else "（无）"


def report_tool_catalog(runtime: AgentRuntime) -> str:
    lines: list[str] = []
    for name in runtime.report_registry.list_names():
        tool = runtime.report_registry.get(name)
        if tool:
            lines.append(f"- {tool.name}: {tool.description}")
    return "\n".join(lines) if lines else "（无）"


def _coerce_bool(val: Any, default: bool) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off", ""):
            return False
    return default


def apply_plan_flags(
    data: dict[str, Any],
    query: str,
    report_mode: bool,
) -> tuple[str, str, NodeEnableFlags]:
    flags = NodeEnableFlags(
        enable_knowledge_retrieve=_coerce_bool(data.get("enable_knowledge_retrieve"), True),
        enable_data_retrieve=_coerce_bool(data.get("enable_data_retrieve"), False),
        enable_process=_coerce_bool(data.get("enable_process"), False),
        enable_chart=_coerce_bool(data.get("enable_chart"), False),
        enable_report=_coerce_bool(data.get("enable_report"), report_mode),
    )
    text_q = str(data.get("text_query", "") or "").strip()
    data_q = str(data.get("data_query", "") or "").strip()
    text_q, data_q = normalize_queries_for_flags(text_q, data_q, flags, fallback=query)

    return text_q, data_q, flags
