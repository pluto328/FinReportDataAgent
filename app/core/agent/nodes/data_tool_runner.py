"""Shared data tool execution (used by data_tool node and process_executor)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.agent.events import emit_tool_end, emit_tool_start
from app.core.agent.nodes._helpers import (
    load_preview_rows_from_path,
    normalize_data_tool_params,
    normalize_file_previews,
    strip_preview_from_tool_result,
    upsert_file_preview,
)
from app.core.agent.state import AgentRuntime, AgentState
from app.core.session.process_artifact_store import register_session_artifacts
from app.schemas.structured import ChartArtifact, DataToolStepResult, FilePreviewEntry, ProcessedDataRef

_PREVIEW_TOOLS = frozenset({"preview_read", "data_filter", "sql_execute", "pandas_execute"})
_PATH_TOOLS = frozenset({"data_filter", "sql_execute", "pandas_execute", "make_chart"})


def _merge_preview_read_previews(
    previews: dict[str, FilePreviewEntry],
    params: dict,
    result: dict,
) -> dict[str, FilePreviewEntry]:
    from app.core.tools.structured_ops import normalize_file_paths

    paths = normalize_file_paths(
        result.get("path") or params.get("file_path"),
        file_paths=result.get("paths") or params.get("file_paths"),
    )
    merged = dict(previews)
    for path in paths:
        merged = upsert_file_preview(
            merged,
            file_name=Path(path).name,
            path=path,
            description="原始数据预览",
            preview_rows=load_preview_rows_from_path(path),
        )
    return merged


def _merge_previews(
    previews: dict[str, FilePreviewEntry],
    tool_name: str,
    params: dict,
    result: dict,
) -> dict[str, FilePreviewEntry]:
    if tool_name not in _PREVIEW_TOOLS:
        return previews
    if tool_name == "preview_read":
        return _merge_preview_read_previews(previews, params, result)
    path = str(result.get("path") or "").strip()
    if not path:
        return previews
    desc = str(params.get("artifact_description") or params.get("description") or "").strip()
    rows = load_preview_rows_from_path(path)
    return upsert_file_preview(
        previews,
        file_name=Path(path).name,
        path=path,
        description=desc,
        preview_rows=rows,
    )


async def execute_data_tool(
    state: AgentState,
    runtime: AgentRuntime,
    *,
    tool_name: str,
    params: dict[str, Any],
    step_no: int,
    emit_events: bool = True,
) -> dict[str, Any]:
    """Run one data tool; return patch fields (data_tool_steps, file_previews, etc.)."""
    session_id = state.get("session_id", "default")
    normalized = normalize_data_tool_params(
        dict(params),
        file_paths=list(state.get("data_file_paths") or []),
        tool_name=tool_name,
        session_id=session_id,
        settings=runtime.settings,
        extra_paths=list(state.get("processed_data_refs") or []),
    )
    file_path = str(normalized.get("file_path") or "")
    if tool_name in {"sql_execute", "pandas_execute", "preview_read"}:
        file_path = ", ".join(normalized.get("file_paths") or ([file_path] if file_path else []))

    if emit_events:
        await emit_tool_start("data", tool_name)

    tool = runtime.data_registry.get(tool_name)
    error = ""
    result: dict = {}
    artifact_ref: ProcessedDataRef | None = None
    chart_item: ChartArtifact | None = None
    catalog_updates: dict[str, str] = {}
    file_previews = dict(normalize_file_previews(state.get("file_previews")))

    if tool:
        try:
            run_params = dict(normalized)
            run_params.setdefault("session_id", session_id)
            run_params.setdefault("settings", runtime.settings)
            raw = await tool.run(**run_params)
            result = raw if isinstance(raw, dict) else {"result": raw}
            if result.get("error"):
                error = str(result["error"])
            elif tool_name == "preview_read":
                file_previews = _merge_previews(file_previews, tool_name, normalized, result)
            elif tool_name == "make_chart" and result.get("path"):
                saved = str(result["path"])
                chart_item = ChartArtifact(
                    path=saved,
                    description=str(normalized.get("artifact_description", normalized.get("title", ""))),
                    title=str(normalized.get("title", "")),
                    chart_type=str(normalized.get("chart_type", "table")),
                )
                desc = str(normalized.get("artifact_description", normalized.get("description", "图表")))
                catalog_updates[saved] = desc
                artifact_ref = ProcessedDataRef(path=saved, preview="", mode="tool", source_file=file_path)
            elif tool_name in _PATH_TOOLS and result.get("path"):
                saved = str(result["path"])
                desc = str(normalized.get("artifact_description", normalized.get("description", tool_name)))
                catalog_updates[saved] = desc
                artifact_ref = ProcessedDataRef(
                    path=saved,
                    preview="",
                    mode="tool",
                    source_file=file_path,
                )
                file_previews = _merge_previews(file_previews, tool_name, normalized, result)
        except Exception as exc:
            error = str(exc)
    else:
        error = f"tool not found: {tool_name}"

    stored_result = strip_preview_from_tool_result(result) if not error else {"error": error}
    if not error and tool_name == "preview_read":
        paths = list(result.get("paths") or [])
        if paths:
            stored_result = {
                "paths": paths,
                "path": paths[0],
                "file_names": [Path(p).name for p in paths],
            }

    output_path = str(stored_result.get("path") or "") if stored_result and not error else ""
    output_names = stored_result.get("file_names") if isinstance(stored_result.get("file_names"), list) else []
    if emit_events:
        await emit_tool_end(
            "data",
            tool_name,
            ok=not error,
            error=error,
            output_filename=",".join(output_names) if output_names else (Path(output_path).name if output_path else ""),
        )

    if catalog_updates and not error:
        register_session_artifacts(session_id, catalog_updates, runtime.settings)

    step = DataToolStepResult(
        step=step_no,
        tool_name=tool_name,
        params=normalized,
        result=stored_result if not error else {"error": error},
        error=error,
        artifact_ref=artifact_ref,
    )
    patch: dict[str, Any] = {
        "data_tool_steps": [step],
        "file_previews": file_previews,
    }
    if artifact_ref:
        patch["processed_data_refs"] = [artifact_ref.path]
    if chart_item:
        patch["chart_artifacts"] = [chart_item]
    patch["success"] = not error
    patch["error"] = error
    return patch
