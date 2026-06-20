"""Shared helpers for LangGraph agent nodes."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.core.agent.state import AgentRuntime, AgentState
from app.schemas.structured import (
    DataToolStepResult,
    NodeEnableFlags,
    PlanStepResult,
    ReportStepResult,
)


def append_node(state: AgentState, name: str) -> dict:
    return {"nodes_traversed": [name]}


def parse_llm_json(raw: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", raw, re.S)
    return json.loads(match.group(0) if match else raw)


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


def format_data_tool_history(steps: list[DataToolStepResult]) -> str:
    if not steps:
        return "（尚无数据处理工具执行记录）"
    lines: list[str] = []
    for s in steps:
        payload = s.result if not s.error else {"error": s.error}
        lines.append(
            f"步骤{s.step} {s.tool_name}({json.dumps(s.params, ensure_ascii=False)}) "
            f"-> {json.dumps(payload, ensure_ascii=False)[:800]}"
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


def apply_plan_flags(
    data: dict[str, Any],
    query: str,
    report_mode: bool,
) -> tuple[str, str, NodeEnableFlags, str]:
    text_q = data.get("text_query") or query
    data_q = data.get("data_query") or query
    flags = NodeEnableFlags(
        enable_knowledge_retrieve=bool(data.get("enable_knowledge_retrieve", True)),
        enable_data_retrieve=bool(data.get("enable_data_retrieve", False)),
        enable_process=bool(data.get("enable_process", False)),
        enable_chart=bool(data.get("enable_chart", False)),
        enable_report=bool(data.get("enable_report", report_mode)),
    )
    description = str(
        data.get("data_process_plan", "")
        or data.get("dataprocessplan", "")
        or data.get("data_process_description", "")
        or data.get("data_process_flow", "")
    )
    return text_q, data_q, flags, description
