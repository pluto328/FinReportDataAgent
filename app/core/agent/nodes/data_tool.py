"""data_tool node — execute data_registry tools."""

from __future__ import annotations

import asyncio

from app.core.agent.events import emit_tool_end, emit_tool_start
from app.core.agent.nodes._debug_runtime import print_node_result, sample_state, stub_runtime
from app.core.agent.nodes._helpers import append_node, summarize_data_tool_steps
from app.core.agent.nodes._node_log import node_logger
from app.core.agent.state import AgentRuntime, AgentState
from app.core.session.process_artifact_store import register_session_artifacts
from app.schemas.structured import ChartArtifact, DataToolStepResult, PendingToolCall, ProcessedDataRef


def _path_tools() -> set[str]:
    return {"data_filter", "sql_execute", "pandas_execute", "make_chart"}


async def data_tool_node(state: AgentState, runtime: AgentRuntime) -> dict:
    log = node_logger(runtime.settings, "data_tool")
    pending = state.get("pending_tool")
    if not pending or pending.phase != "data" or not pending.tool_name:
        log.start(plan_action="skip")
        log.info("无待执行 data tool，跳过")
        log.end(process_done=True)
        return {**append_node(state, "data_tool"), "process_done": True, "pending_tool": None}

    session_id = state.get("session_id", "default")
    step_no = state.get("process_step", 0) + 1
    params = dict(pending.params)
    file_path = pending.file_path or params.get("file_path", "")
    params.setdefault("file_path", file_path)
    tool_name = pending.tool_name
    log.start(tool_name=tool_name, step=step_no, file_path=file_path, params=params)
    log.info("触发 data tool 调用", tool_name=tool_name)
    await emit_tool_start("data", tool_name)

    tool = runtime.data_registry.get(tool_name)
    error = ""
    result: dict = {}
    artifact_ref: ProcessedDataRef | None = None
    chart_item: ChartArtifact | None = None
    catalog_updates: dict[str, str] = {}

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
                log.info("预览读取成功", rows=len(result.get("preview") or []))
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
            elif tool_name in _path_tools() and result.get("path"):
                saved = str(result["path"])
                desc = str(params.get("artifact_description", params.get("description", tool_name)))
                catalog_updates[saved] = desc
                artifact_ref = ProcessedDataRef(path=saved, preview="", mode="tool", source_file=file_path)
                log.info("已保存处理结果", path=saved)
            else:
                log.info("工具调用成功", tool_name=tool_name)
        except Exception as exc:
            error = str(exc)
            log.fail("工具调用异常", tool_name=tool_name, error=error)
    else:
        error = f"tool not found: {tool_name}"
        log.fail("工具未找到", tool_name=tool_name)

    await emit_tool_end("data", tool_name, ok=not error, error=error)

    if catalog_updates and not error:
        register_session_artifacts(session_id, catalog_updates, runtime.settings)

    step = DataToolStepResult(
        step=step_no,
        tool_name=tool_name,
        params=params,
        result=result if not error else {"error": error},
        error=error,
        artifact_ref=artifact_ref,
    )
    prior_steps = state.get("data_tool_steps") or []
    all_steps = [*prior_steps, step]
    summary = summarize_data_tool_steps(all_steps)

    out: dict = {
        "data_tool_steps": [step],
        "process_step": step_no,
        "process_result": summary or result,
        "process_done": False,
        "pending_tool": None,
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
