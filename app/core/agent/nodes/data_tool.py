"""data_tool node — execute data_registry tools."""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.core.agent.events import emit_tool_end, emit_tool_start
from app.core.agent.nodes._debug_runtime import print_node_result, sample_state, stub_runtime
from app.core.agent.nodes._helpers import (
    append_node,
    load_preview_rows_from_path,
    normalize_data_tool_params,
    normalize_file_previews,
    strip_preview_from_tool_result,
    summarize_data_tool_steps,
    upsert_file_preview,
)
from app.core.agent.nodes._node_log import node_logger
from app.core.agent.state import AgentRuntime, AgentState
from app.core.session.process_artifact_store import register_session_artifacts
from app.schemas.structured import ChartArtifact, DataToolStepResult, FilePreviewEntry, PendingToolCall, ProcessedDataRef

_PREVIEW_TOOLS = frozenset({"preview_read", "data_filter", "sql_execute", "pandas_execute"})
_PATH_TOOLS = frozenset({"data_filter", "sql_execute", "pandas_execute", "make_chart"})


def _preview_target(
    tool_name: str,
    params: dict,
    result: dict,
) -> tuple[str, str, str]:
    """Return (file_name, absolute_path, description) for preview cache."""
    desc = str(params.get("artifact_description") or params.get("description") or "").strip()
    if tool_name == "preview_read":
        path = str(params.get("file_path") or result.get("path") or "").strip()
        return Path(path).name, path, desc or "原始数据预览"
    path = str(result.get("path") or "").strip()
    return Path(path).name, path, desc


def _merge_preview_read_previews(
    state: AgentState,
    params: dict,
    result: dict,
) -> dict[str, FilePreviewEntry]:
    from app.core.tools.structured_ops import normalize_file_paths

    paths = normalize_file_paths(
        result.get("path") or params.get("file_path"),
        file_paths=result.get("paths") or params.get("file_paths"),
    )
    merged = normalize_file_previews(state.get("file_previews"))
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
    state: AgentState,
    tool_name: str,
    params: dict,
    result: dict,
) -> dict[str, FilePreviewEntry]:
    if tool_name not in _PREVIEW_TOOLS:
        return normalize_file_previews(state.get("file_previews"))
    if tool_name == "preview_read":
        return _merge_preview_read_previews(state, params, result)
    file_name, path, desc = _preview_target(tool_name, params, result)
    if not path:
        return normalize_file_previews(state.get("file_previews"))
    rows = load_preview_rows_from_path(path)
    return upsert_file_preview(
        state.get("file_previews"),
        file_name=file_name,
        path=path,
        description=desc,
        preview_rows=rows,
    )


async def data_tool_node(state: AgentState, runtime: AgentRuntime) -> dict:
    log = node_logger(runtime.settings, "data_tool")
    pending = state.get("pending_tool")
    if not pending or pending.phase != "data" or not pending.tool_name:
        log.start(plan_action="skip")
        log.info("无待执行 data tool，跳过")
        log.end(process_done=False)
        return {**append_node(state, "data_tool"), "process_done": False, "pending_tool": None}

    session_id = state.get("session_id", "default")
    step_no = state.get("process_step", 0) + 1
    tool_name = pending.tool_name
    params = normalize_data_tool_params(
        dict(pending.params),
        file_paths=list(state.get("data_file_paths") or []),
        tool_name=tool_name,
        session_id=session_id,
        settings=runtime.settings,
        extra_paths=list(state.get("processed_data_refs") or []),
    )
    file_path = pending.file_path or params.get("file_path", "")
    if tool_name in {"sql_execute", "pandas_execute", "preview_read"}:
        file_path = ", ".join(params.get("file_paths") or ([file_path] if file_path else []))
    log.start(tool_name=tool_name, step=step_no, file_path=file_path, params=params)
    log.info("触发 data tool 调用", tool_name=tool_name)
    await emit_tool_start("data", tool_name)

    tool = runtime.data_registry.get(tool_name)
    error = ""
    result: dict = {}
    artifact_ref: ProcessedDataRef | None = None
    chart_item: ChartArtifact | None = None
    catalog_updates: dict[str, str] = {}
    file_previews = dict(state.get("file_previews") or {})

    if tool:
        try:
            run_params = dict(params)
            run_params.setdefault("session_id", session_id)
            run_params.setdefault("settings", runtime.settings)
            raw = await tool.run(**run_params)
            result = raw if isinstance(raw, dict) else {"result": raw}

            if result.get("error"):
                error = str(result["error"])
                log.fail("工具返回错误", tool_name=tool_name, error=error)
            elif tool_name == "preview_read":
                file_previews = _merge_previews(state, tool_name, params, result)
                paths = result.get("paths") or []
                log.info("预览读取成功", files=len(paths))
            elif tool_name == "make_chart" and result.get("path"):
                saved = str(result["path"])
                chart_item = ChartArtifact(
                    path=saved,
                    description=str(params.get("description", params.get("title", ""))),
                    title=str(params.get("title", "")),
                    chart_type=str(params.get("chart_type", "table")),
                )
                desc = str(params.get("artifact_description", params.get("description", "图表")))
                catalog_updates[saved] = desc
                artifact_ref = ProcessedDataRef(path=saved, preview="", mode="tool", source_file=file_path)
                log.info("图表生成成功", path=saved)
            elif tool_name in _PATH_TOOLS and result.get("path"):
                saved = str(result["path"])
                desc = str(params.get("artifact_description", params.get("description", tool_name)))
                catalog_updates[saved] = desc
                artifact_ref = ProcessedDataRef(
                    path=saved,
                    preview="",
                    mode="tool",
                    source_file=file_path,
                )
                file_previews = _merge_previews(state, tool_name, params, result)
                log.info("已保存处理结果", path=saved)
            else:
                log.info("工具调用成功", tool_name=tool_name)
        except Exception as exc:
            error = str(exc)
            log.fail("工具调用异常", tool_name=tool_name, error=error)
    else:
        error = f"tool not found: {tool_name}"
        log.fail("工具未找到", tool_name=tool_name)

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
        params=params,
        result=stored_result if not error else {"error": error},
        error=error,
        artifact_ref=artifact_ref,
    )
    prior_steps = state.get("data_tool_steps") or []
    all_steps = [*prior_steps, step]
    summary = summarize_data_tool_steps(all_steps)

    out: dict = {
        "data_tool_steps": [step],
        "process_step": step_no,
        "process_result": summary or stored_result,
        "process_done": False,
        "pending_tool": None,
        "file_previews": file_previews,
        **append_node(state, "data_tool"),
    }
    if artifact_ref:
        out["processed_data"] = [artifact_ref]
        out["processed_data_refs"] = [artifact_ref.path]
    if chart_item:
        out["chart_artifacts"] = [chart_item]
    log.end(tool_name=tool_name, step=step_no, success=not error, next_node="data_processor")
    return out


async def debug_data_tool_node() -> None:
    state = sample_state(
        pending_tool=PendingToolCall(
            phase="data",
            tool_name="preview_read",
            file_path="data/raw_structured/sample.csv",
            params={},
        ),
        data_file_paths=["data/raw_structured/sample.csv"],
    )
    runtime = stub_runtime()
    result = await data_tool_node(state, runtime)
    print_node_result("data_tool_node", result)


if __name__ == "__main__":
    asyncio.run(debug_data_tool_node())
