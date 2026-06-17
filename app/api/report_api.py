"""Report generation and download API."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from app.core.agent.runner import sse_agent_events
from app.dependencies import AppContainer, get_app_container
from app.schemas.query import ReportRequest, SearchRequest

router = APIRouter()


@router.post("/stream")
async def generate_report_stream(
    body: ReportRequest,
    container: AppContainer = Depends(get_app_container),
) -> StreamingResponse:
    search_req = SearchRequest(
        query=body.query,
        session_id=body.session_id,
        chat_history=body.chat_history,
        report_mode=True,
    )
    return StreamingResponse(
        sse_agent_events(container, search_req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{session_id}/download")
async def download_report(
    session_id: str,
    container: AppContainer = Depends(get_app_container),
) -> FileResponse:
    path = container.settings.report_output_path / session_id / "report.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail="report not found")
    return FileResponse(path, filename=f"report_{session_id}.md", media_type="text/markdown")
