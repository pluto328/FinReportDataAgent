"""Agent runner helper."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from app.core.agent.events import ProgressEmitter, reset_emitter, set_emitter
from app.core.agent.llm_capture import finalize_capture, start_capture
from app.core.agent.state import AgentState
from app.core.session.process_artifact_store import reset_session_workspace
from app.config.settings import get_settings
from app.dependencies import AppContainer
from app.schemas.query import SearchRequest, SearchResponse

AGENT_RECURSION_LIMIT = 50


def _build_initial_state(request: SearchRequest) -> AgentState:
    session_id = request.session_id or ""
    return {
        "user_query": (request.query or "").strip(),
        "user_require": "",
        "after_reporter_retrieval_goto": "",
        "chat_history": request.chat_history,
        "report_mode": request.report_mode,
        "session_id": session_id,
        "knowledge_chunks": [],
        "meta_hits": [],
        "data_file_paths": [],
        "processed_data_refs": [],
        "chart_artifacts": [],
        "nodes_traversed": [],
        "plan_steps": [],
        "plan_step": 0,
        "plan_done": False,
        "pending_tool": None,
        "data_process_plan": "",
        "process_steps_plan": [],
        "pending_chart_params": None,
        "process_repair_attempted": False,
        "worker_step": None,
        "worker_index": None,
        "data_tool_steps": [],
        "file_previews": {},
        "process_step": 0,
        "process_done": False,
        "report_steps": [],
        "report_step": 0,
        "report_done": False,
        "report_context": {},
    }


def _final_to_response(final: AgentState, request: SearchRequest) -> SearchResponse:
    charts = final.get("chart_artifacts") or []
    chart_paths = [c.path if hasattr(c, "path") else str(c) for c in charts]
    answer = final.get("final_answer", "")
    return SearchResponse(
        answer=answer,
        message=answer,
        session_id=final.get("session_id", request.session_id),
        status=final.get("status", "ok"),
        knowledge_chunks=final.get("knowledge_chunks") or [],
        meta_hits=final.get("meta_hits") or [],
        chart_artifacts=chart_paths,
    )


def _response_to_done_payload(response: SearchResponse) -> dict[str, Any]:
    return {
        "type": "done",
        "answer": response.answer,
        "message": response.message,
        "session_id": response.session_id,
        "status": response.status,
        "knowledge_chunks": [c.model_dump() for c in response.knowledge_chunks],
        "meta_hits": [m.model_dump() for m in response.meta_hits],
        "chart_artifacts": response.chart_artifacts,
    }


async def run_agent_stream(
    container: AppContainer,
    request: SearchRequest,
    emitter: ProgressEmitter,
) -> SearchResponse:
    graph = container.agent_graph
    session_id = request.session_id or ""
    if request.new_session and session_id:
        reset_session_workspace(session_id, container.settings)
    capture_path = start_capture(request.query or "")
    initial = _build_initial_state(request)
    token = set_emitter(emitter)
    try:
        final = await graph.ainvoke(
            initial,
            config={"recursion_limit": AGENT_RECURSION_LIMIT},
        )
    finally:
        reset_emitter(token)
        saved = finalize_capture()
        if saved:
            await emitter.emit({"type": "llm_capture_saved", "path": str(saved)})
    response = _final_to_response(final, request)
    await emitter.emit(_response_to_done_payload(response))
    return response


async def sse_agent_events(
    container: AppContainer,
    request: SearchRequest,
) -> AsyncIterator[str]:
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    emitter = ProgressEmitter(queue)

    async def _run() -> None:
        try:
            await run_agent_stream(container, request, emitter)
        except Exception as exc:
            from app.common.logger import logger

            logger.exception("agent stream failed")
            await emitter.emit({"type": "error", "message": str(exc)})
        finally:
            await queue.put(None)

    task = asyncio.create_task(_run())
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=20.0)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            if event is None:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
