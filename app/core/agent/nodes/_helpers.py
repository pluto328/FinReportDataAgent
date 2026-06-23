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
    DataToolStepResult,
    FilePreviewEntry,
    NodeEnableFlags,
    PlanStepResult,
    ReportStepResult,
)

if TYPE_CHECKING:
    from app.config.settings import Settings
else:
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
    from app.core.tools.artifact_utils import json_safe

    cleaned: list[dict[str, Any]] = []
    for item in rows[:FILE_PREVIEW_STORE_ROWS]:
        if isinstance(item, dict):
            cleaned.append(json_safe(item))
    return cleaned


def load_preview_rows_from_path(path: str, *, rows: int = FILE_PREVIEW_STORE_ROWS) -> list[dict[str, Any]]:
    from app.core.tools.artifact_utils import preview_dataframe_rows
    from app.core.tools.structured_ops import read_table_preview

    text = str(path or "").strip()
    if not text:
        return []
    try:
        df = read_table_preview(text, rows=rows)
        return preview_dataframe_rows(df, rows=rows)
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


def collect_processed_artifact_paths(
    state: AgentState,
    settings: Settings | None = None,
) -> list[tuple[str, str]]:
    """Processed artifact paths (catalog / refs / tool outputs), excluding raw previews."""
    from app.core.session.process_artifact_store import get_session_catalog, resolve_catalog_path

    s = settings or get_settings()
    session_id = state.get("session_id", "")
    extra = [str(p) for p in (state.get("data_file_paths") or []) if p]
    extra.extend(str(p) for p in (state.get("processed_data_refs") or []) if p)
    seen: set[str] = set()
    out: list[tuple[str, str]] = []

    def add(raw: str, desc: str = "") -> None:
        text = str(raw or "").strip()
        if not text:
            return
        resolved = resolve_catalog_path(session_id, text, s, extra_paths=extra) or text
        if resolved in seen or not Path(resolved).exists():
            return
        seen.add(resolved)
        out.append((resolved, desc))

    for path, desc in get_session_catalog(session_id, s).items():
        add(path, desc)
    for step in state.get("data_tool_steps") or []:
        if step.error:
            continue
        artifact = step.artifact_ref
        if artifact and artifact.path:
            add(artifact.path)
    return out


def format_processed_data_full_for_prompt(
    state: AgentState,
    settings: Settings | None = None,
) -> tuple[str, list[str]]:
    """Load full processed artifact text for reporter; truncate by context_size_threshold_chars."""
    from app.core.tools.artifact_utils import load_artifact_text

    s = settings or get_settings()
    items = collect_processed_artifact_paths(state, s)
    if not items:
        return "（无已处理数据产物）", []
    max_total = s.context_size_threshold_chars
    per_file = max(max_total // len(items), 1000)
    blocks: list[str] = []
    loaded_paths: list[str] = []
    used = 0
    for path, desc in items:
        remaining = max_total - used
        if remaining <= 0:
            blocks.append(
                f"#### {Path(path).name}\n（已达上下文字符上限 {max_total}，后续文件已省略）"
            )
            break
        cap = min(per_file, remaining)
        try:
            text = load_artifact_text(path, max_chars=cap)
            truncated = len(text) >= cap
            name = Path(path).name
            label = f"{name} ({desc})" if desc else name
            if truncated:
                label += f" [已截断至 {cap} 字符]"
            blocks.append(f"#### {label}\n{text}")
            used += len(text)
            loaded_paths.append(path)
        except OSError as exc:
            blocks.append(f"#### {Path(path).name}: 读取失败 {exc}")
    return "\n\n".join(blocks), loaded_paths


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


def is_one_shot_mode(settings: Settings | None = None, state: AgentState | None = None) -> bool:
    if settings is not None:
        return getattr(settings, "agent_process_mode", "one_shot") == "one_shot"
    if state is not None:
        from app.config.settings import get_settings

        return get_settings().agent_process_mode == "one_shot"
    return True


def process_entry_node(state: AgentState) -> str:
    from app.config.settings import get_settings

    if is_one_shot_mode(get_settings()):
        return "process_planner"
    return "data_processor"


def load_file_previews_for_paths(
    paths: list[str],
    existing: dict[str, FilePreviewEntry | dict[str, Any]] | None = None,
) -> dict[str, FilePreviewEntry]:
    merged = normalize_file_previews(existing)
    for path in paths:
        text = str(path or "").strip()
        if not text:
            continue
        name = Path(text).name
        if name in merged:
            continue
        rows = load_preview_rows_from_path(text)
        merged = upsert_file_preview(
            merged,
            file_name=name,
            path=text,
            description="原始数据预览",
            preview_rows=rows,
        )
    return merged


def format_meta_columns_for_prompt(meta: list, *, limit: int = 8) -> str:
    if not meta:
        return "（暂无元数据列信息）"
    lines: list[str] = []
    for m in meta[:limit]:
        name = m.record.file_name or Path(str(m.record.file_path or "")).name
        text = (m.record.search_text or "")[:300]
        lines.append(f"- {name}: {text}")
    return "\n".join(lines)


def expected_tools_from_plan(
    plan: str,
    *,
    enable_chart: bool,
    skip_preview: bool,
) -> set[str]:
    tools: set[str] = set()
    text = str(plan or "")
    lower = text.lower()
    if not skip_preview and ("预览" in text or "preview" in lower):
        tools.add("preview_read")
    if any(k in lower or k in text for k in ("pandas", "排序", "筛选", "提取", "汇总")):
        tools.add("pandas_execute")
    if "sql" in lower:
        tools.add("sql_execute")
    if enable_chart and any(k in text for k in ("图", "chart", "绘制", "柱状")):
        tools.add("make_chart")
    if not tools and text.strip():
        tools.add("pandas_execute")
    return tools


def completed_tool_names(steps: list[DataToolStepResult]) -> set[str]:
    return {s.tool_name for s in steps if not s.error}


def is_data_process_plan_complete(state: AgentState) -> bool:
    plan = str(state.get("data_process_plan") or "").strip()
    if not plan:
        steps_plan = state.get("process_steps_plan") or []
        if steps_plan:
            executed = completed_tool_names(state.get("data_tool_steps") or [])
            planned = {str(s.get("tool", "")) for s in steps_plan if s.get("tool")}
            chart_pending = state.get("pending_chart_params")
            flags = state.get("node_flags")
            if chart_pending and flags and flags.enable_chart:
                planned.add("make_chart")
            return planned.issubset(executed)
        # React loop (data_tool): without a plan, wait for LLM action=done.
        return False
    flags = state.get("node_flags")
    enable_chart = bool(flags and flags.enable_chart)
    previews = state.get("file_previews") or {}
    file_paths = state.get("data_file_paths") or []
    skip_preview = bool(file_paths) and all(Path(p).name in previews for p in file_paths if p)
    expected = expected_tools_from_plan(plan, enable_chart=enable_chart, skip_preview=skip_preview)
    done = completed_tool_names(state.get("data_tool_steps") or [])
    return expected.issubset(done)


def reporter_has_preloaded_data(state: AgentState) -> bool:
    from app.config.settings import get_settings

    if collect_processed_artifact_paths(state, get_settings()):
        return True
    steps = state.get("data_tool_steps") or []
    return any(
        not s.error and s.tool_name in {"pandas_execute", "sql_execute", "data_filter"}
        for s in steps
    )


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
