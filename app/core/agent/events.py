"""Agent progress events for SSE streaming."""

from __future__ import annotations

import json
import re
from contextvars import ContextVar, Token
from typing import Any, Literal

from app.infrastructure.llm_client import LLMClient

_emitter_var: ContextVar[ProgressEmitter | None] = ContextVar("agent_progress_emitter", default=None)

_FIELD_START_RE = re.compile(r'"([^"]+)"\s*:\s*"', re.DOTALL)

_DATA_TOOL_WAIT_LABELS: dict[str, str] = {
    "preview_read": "正在预览数据",
    "data_filter": "正在筛选数据",
    "pandas_execute": "正在处理数据",
    "sql_execute": "正在查询数据",
    "make_chart": "正在绘图",
    "read_data_file": "正在读取数据",
}


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


class _JsonStringFieldExtractor:
    """Incrementally extract a JSON string field value from streamed LLM output."""

    def __init__(self, field: str) -> None:
        self._field = field
        self._carry = ""
        self._in_value = False
        self._done = False
        self._escaped = False
        self._unicode_left = 0
        self._unicode_hex = ""
        self._value = ""

    def feed(self, chunk: str) -> str:
        if self._done or not chunk:
            return ""
        delta_parts: list[str] = []
        for ch in chunk:
            if self._done:
                break
            if not self._in_value:
                self._carry += ch
                match = _FIELD_START_RE.search(self._carry)
                if match and match.group(1) == self._field:
                    self._in_value = True
                    remainder = self._carry[match.end() :]
                    self._carry = ""
                    if remainder:
                        delta_parts.append(self._feed_value_characters(remainder))
                elif len(self._carry) > len(self._field) + 24:
                    self._carry = self._carry[-(len(self._field) + 24) :]
                continue
            part = self._feed_value_characters(ch)
            if part:
                delta_parts.append(part)
        return "".join(delta_parts)

    def _feed_value_characters(self, text: str) -> str:
        delta_parts: list[str] = []
        for ch in text:
            if self._done:
                break
            if self._unicode_left:
                self._unicode_hex += ch
                self._unicode_left -= 1
                if self._unicode_left == 0:
                    try:
                        decoded = chr(int(self._unicode_hex, 16))
                    except ValueError:
                        decoded = ""
                    self._value += decoded
                    delta_parts.append(decoded)
                    self._unicode_hex = ""
                continue
            if self._escaped:
                self._escaped = False
                if ch == "u":
                    self._unicode_left = 4
                    self._unicode_hex = ""
                else:
                    mapping = {"n": "\n", "r": "\r", "t": "\t", '"': '"', "\\": "\\", "/": "/"}
                    decoded = mapping.get(ch, ch)
                    self._value += decoded
                    delta_parts.append(decoded)
                continue
            if ch == "\\":
                self._escaped = True
                continue
            if ch == '"':
                self._done = True
                break
            self._value += ch
            delta_parts.append(ch)
        return "".join(delta_parts)

    @property
    def value(self) -> str:
        return self._value

    @property
    def done(self) -> bool:
        return self._done


async def emit_node_start(node: str) -> None:
    emitter = get_emitter()
    if emitter:
        await emitter.emit({"type": "node_start", "node": node})


async def emit_retrieval_status(*, kind: str, query: str) -> None:
    if not query.strip():
        return
    q = query.strip()
    if kind == "text":
        label = f"正在检索 {q} 相关文本"
    else:
        label = f"正在检索 {q} 相关数据"
    await emit_progress_waiting(label, active=True)


async def emit_progress_waiting(text: str = "", *, active: bool = True) -> None:
    emitter = get_emitter()
    if not emitter:
        return
    await emitter.emit({"type": "progress_waiting", "text": text, "active": active})


async def emit_progress_line(text: str) -> None:
    emitter = get_emitter()
    if emitter and text:
        await emitter.emit({"type": "progress_line", "text": text})


async def emit_progress_delta(text: str) -> None:
    emitter = get_emitter()
    if emitter and text:
        await emitter.emit({"type": "progress_delta", "text": text})


async def emit_tool_start(phase: str, tool_name: str) -> None:
    emitter = get_emitter()
    if not emitter:
        return
    if phase == "data":
        label = _DATA_TOOL_WAIT_LABELS.get(tool_name, f"正在执行 {tool_name}")
        await emit_progress_waiting(label, active=True)
        return
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


async def emit_tool_end(
    phase: str,
    tool_name: str,
    *,
    ok: bool,
    error: str = "",
    output_filename: str = "",
) -> None:
    emitter = get_emitter()
    if not emitter:
        return
    if phase == "data":
        await emit_progress_waiting(active=False)
        if ok and output_filename:
            await emit_progress_line(f"已生成：{output_filename}")
        elif not ok:
            await emit_progress_line(f"{tool_name} 失败：{error[:120]}" if error else f"{tool_name} 失败")
        return
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
    stream_field: str | None = None,
    stream_as: Literal["progress", "answer"] = "progress",
    emit_thinking: bool = True,
) -> str:
    emitter = get_emitter()
    parts: list[str] = []
    field_extractor = _JsonStringFieldExtractor(stream_field) if stream_field else None
    if emitter:
        if field_extractor and stream_as == "progress":
            await emitter.emit({"type": "planning_start", "phase": phase})
        elif field_extractor and stream_as == "answer":
            await emitter.emit({"type": "answer_start"})
        elif emit_thinking:
            await emitter.emit({"type": "thinking_start", "phase": phase, "purpose": purpose})
    async for chunk in llm.astream(prompt):
        parts.append(chunk)
        if emitter and field_extractor:
            delta = field_extractor.feed(chunk)
            if delta:
                if stream_as == "answer":
                    await emitter.emit({"type": "answer_delta", "text": delta})
                else:
                    await emitter.emit({"type": "progress_delta", "text": delta})
        elif emitter and emit_thinking:
            await emitter.emit({"type": "thinking_delta", "phase": phase, "text": chunk})
    raw = "".join(parts)
    parsed: dict[str, Any] | None = None
    try:
        from app.core.agent.nodes._helpers import parse_llm_json

        parsed = parse_llm_json(raw)
    except (json.JSONDecodeError, ValueError, AttributeError):
        parsed = None
    if emitter:
        if field_extractor and stream_as == "answer":
            answer_text = field_extractor.value
            if parsed is not None and not answer_text:
                answer_text = str(parsed.get(stream_field or "") or "")
            await emitter.emit({"type": "answer_end", "text": answer_text})
        elif field_extractor and stream_as == "progress":
            planning_thought = field_extractor.value
            if parsed is not None and not planning_thought:
                planning_thought = str(parsed.get(stream_field or "") or "")
            await emitter.emit(
                {
                    "type": "planning_end",
                    "phase": phase,
                    "planning_thought": planning_thought,
                }
            )
            await emit_progress_line("规划完成")
        elif emit_thinking:
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
