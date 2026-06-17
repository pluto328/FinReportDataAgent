"""Knowledge base search API (SSE streaming)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.core.agent.runner import sse_agent_events
from app.dependencies import AppContainer, get_app_container
from app.schemas.query import SearchRequest

router = APIRouter()


@router.post("/stream")
async def search_stream(
    body: SearchRequest,
    container: AppContainer = Depends(get_app_container),
) -> StreamingResponse:
    return StreamingResponse(
        sse_agent_events(container, body),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
