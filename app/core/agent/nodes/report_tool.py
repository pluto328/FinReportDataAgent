"""report_tool node — execute report_registry tools and return to reporter."""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.core.agent.events import emit_tool_end, emit_tool_start
from app.core.agent.nodes._debug_runtime import print_node_result, sample_state, stub_runtime
from app.core.agent.nodes._helpers import append_node, summarize_report_steps
from app.core.agent.nodes._node_log import node_logger
from app.core.agent.state import AgentRuntime, AgentState
from app.core.session.process_artifact_store import get_session_catalog, resolve_catalog_path
from app.schemas.structured import PendingToolCall, ReportStepResult


async def report_tool_node(state: AgentState, runtime: AgentRuntime) -> dict:
    log = node_logger(runtime.settings, "report_tool")
    pending = state.get("pending_tool")
    if not pending or pending.phase != "report" or not pending.tool_name:
        log.start(plan_action="skip")
        log.info("无待执行 report tool，跳过")
        log.end(report_done=True)
        return {**append_node(state, "report_tool"), "report_done": True, "pending_tool": None}

    step_no = state.get("report_step", 0) + 1
    params = dict(pending.params)
    tool_name = pending.tool_name
    session_id = state.get("session_id", "")

    if tool_name == "read_data_file":
        extra_paths = list(state.get("data_file_paths") or [])
        extra_paths.extend(state.get("processed_data_refs") or [])
        raw_path = str(params.get("path", ""))
        resolved = resolve_catalog_path(
            session_id, raw_path, runtime.settings, extra_paths=extra_paths
        )
        if resolved:
            params["path"] = resolved
        elif raw_path:
            catalog = get_session_catalog(session_id, runtime.settings)
            names = [Path(p).name for p in catalog] if catalog else []
            extra_names = [Path(p).name for p in extra_paths if p]
            available = ", ".join(dict.fromkeys(names + extra_names)) or "（无）"
            error = f"文件未找到: {raw_path}；可用文件: {available}"
            log.start(tool_name=tool_name, step=step_no, params=params)
            log.fail("report tool 路径解析失败", path=raw_path, available=available)
            await emit_tool_start("report", tool_name)
            await emit_tool_end("report", tool_name, ok=False, error=error)
            step = ReportStepResult(
                step=step_no,
                tool_name=tool_name,
                params=params,
                result={"ok": False, "error": error},
                error=error,
            )
            log.end(tool_name=tool_name, step=step_no, success=False, next_node="reporter")
            return {
                "report_steps": [step],
                "report_step": step_no,
                "report_done": False,
                "pending_tool": None,
                **append_node(state, "report_tool"),
            }

    log.start(tool_name=tool_name, step=step_no, params=params)
    log.info("触发 report tool 调用", tool_name=tool_name)
    await emit_tool_start("report", tool_name)

    tool = runtime.report_registry.get(tool_name)
    error = ""
    if tool:
        try:
            result = await tool.run(**params)
            if isinstance(result, dict) and result.get("error") and not result.get("ok", True):
                error = str(result["error"])
                log.fail("report tool 调用失败", tool_name=tool_name, error=error)
            else:
                log.info("report tool 调用成功", tool_name=tool_name, path=result.get("path") if isinstance(result, dict) else "")
        except Exception as exc:
            result = {}
            error = str(exc)
            log.fail("report tool 调用异常", tool_name=tool_name, error=error)
    else:
        result = {}
        error = f"report tool not found: {tool_name}"
        log.fail("report tool 未找到", tool_name=tool_name)

    await emit_tool_end("report", tool_name, ok=not error, error=error)

    step = ReportStepResult(
        step=step_no,
        tool_name=tool_name,
        params=params,
        result=result if isinstance(result, dict) else {"result": result},
        error=error,
    )
    prior = state.get("report_steps") or []
    all_steps = [*prior, step]
    report_context = dict(state.get("report_context") or {})
    if isinstance(result, dict) and result.get("content"):
        report_context["loaded_content"] = str(result["content"])
        report_context["loaded_path"] = result.get("path", "")

    log.end(tool_name=tool_name, step=step_no, success=not error, next_node="reporter")
    return {
        "report_steps": [step],
        "report_step": step_no,
        "report_context": report_context,
        "report_done": False,
        "pending_tool": None,
        **append_node(state, "report_tool"),
    }


async def debug_report_tool_node() -> None:
    state = sample_state(
        pending_tool=PendingToolCall(
            phase="report",
            tool_name="read_data_file",
            params={"path": "data/parsed_cache/debug/processed/tool_x.json"},
        ),
    )
    runtime = stub_runtime()
    result = await report_tool_node(state, runtime)
    print_node_result("report_tool_node", result)


if __name__ == "__main__":
    asyncio.run(debug_report_tool_node())
