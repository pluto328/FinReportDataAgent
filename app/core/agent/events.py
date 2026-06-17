"""Agent progress events for SSE streaming."""

from __future__ import annotations

import json
from contextvars import ContextVar, Token
from typing import Any

from app.infrastructure.llm_client import LLMClient

_emitter_var: ContextVar[ProgressEmitter | None] = ContextVar("agent_progress_emitter", default=None)


class ProgressEmitter:
    """Request-scoped event sink; pushes dict events to an async queue."""

    def __init__(self, queue: Any) -> None:
        self._queue = queue

    async def emit(self, event: dict[str, Any]) -> None:
        await self._queue.put(event)


def get_emitter() -> ProgressEmitter | None:
    return _emitter_var.get()


def set_emitter(emitter: ProgressEmitter | None) -> Token:
    return _emitter_var.set(emitter)


def reset_emitter(token: Token) -> None:
    _emitter_var.reset(token)


async def emit_node_start(node: str) -> None:
    emitter = get_emitter()
    if emitter:
        await emitter.emit({"type": "node_start", "node": node})


async def emit_tool_start(phase: str, tool_name: str) -> None:
    emitter = get_emitter()
    if emitter:
        labels = {"plan": "规划", "data": "数据处理", "report": "报告"}
        phase_label = labels.get(phase, phase)
        await emitter.emit(
            {
                "type": "tool_start",
                "phase": phase,
                "name": tool_name,
                "label": f"正在调用{phase_label}工具 {tool_name}…",
            }
        )


async def emit_tool_end(phase: str, tool_name: str, *, ok: bool, error: str = "") -> None:
    emitter = get_emitter()
    if emitter:
        status = "调用成功" if ok else "调用失败"
        label = f"{tool_name} {status}"
        if error:
            label += f"：{error[:200]}"
        await emitter.emit(
            {
                "type": "tool_end",
                "phase": phase,
                "name": tool_name,
                "ok": ok,
                "error": error,
                "label": label,
            }
        )


async def invoke_llm_decision(
    llm: LLMClient,
    prompt: str,
    *,
    phase: str,
    purpose: str = "decision",
) -> str:
    emitter = get_emitter()
    parts: list[str] = []
    if emitter:
        await emitter.emit({"type": "thinking_start", "phase": phase, "purpose": purpose})
    async for chunk in llm.astream(prompt):
        parts.append(chunk)
        if emitter:
            await emitter.emit({"type": "thinking_delta", "phase": phase, "text": chunk})
    raw = "".join(parts)
    parsed: dict[str, Any] | None = None
    try:
        from app.core.agent.nodes._helpers import parse_llm_json

        parsed = parse_llm_json(raw)
    except (json.JSONDecodeError, ValueError, AttributeError):
        parsed = None
    if emitter:
        payload: dict[str, Any] = {"type": "thinking_end", "phase": phase, "raw": raw}
        if parsed is not None:
            payload["parsed"] = parsed
        await emitter.emit(payload)
    from app.core.agent.llm_capture import record_llm_call

    record_llm_call(phase=phase, purpose=purpose, prompt=prompt, output=raw)
    return raw


async def invoke_llm_report_output(llm: LLMClient, prompt: str) -> tuple[str, str]:
    """Collect JSON {report, summary} from LLM; stream report body to answer events."""
    emitter = get_emitter()
    parts: list[str] = []
    if emitter:
        await emitter.emit({"type": "thinking_start", "phase": "reporter", "purpose": "answer"})
    async for chunk in llm.astream(prompt):
        parts.append(chunk)
        if emitter:
            await emitter.emit({"type": "thinking_delta", "phase": "reporter", "text": chunk})
    raw = "".join(parts)
    report = raw.strip()
    summary = ""
    try:
        from app.core.agent.nodes._helpers import parse_llm_json

        data = parse_llm_json(raw)
        report = str(data.get("report") or data.get("answer") or report).strip()
        summary = str(data.get("summary", "") or "").strip()
    except (json.JSONDecodeError, ValueError, AttributeError):
        pass
    if emitter:
        await emitter.emit({"type": "thinking_end", "phase": "reporter", "raw": raw})
        await emitter.emit({"type": "answer_start"})
        if report:
            chunk_size = 80
            for i in range(0, len(report), chunk_size):
                await emitter.emit({"type": "answer_delta", "text": report[i : i + chunk_size]})
        await emitter.emit({"type": "answer_end", "text": report})
    from app.core.agent.llm_capture import record_llm_call

    record_llm_call(phase="reporter", purpose="answer", prompt=prompt, output=raw)
    return report, summary


async def invoke_llm_answer(llm: LLMClient, prompt: str) -> str:
    emitter = get_emitter()
    parts: list[str] = []
    if emitter:
        await emitter.emit({"type": "answer_start"})
    async for chunk in llm.astream(prompt):
        parts.append(chunk)
        if emitter:
            await emitter.emit({"type": "answer_delta", "text": chunk})
    answer = "".join(parts)
    if emitter:
        await emitter.emit({"type": "answer_end", "text": answer})
    return answer
