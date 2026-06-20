"""Session lifecycle API."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.session.process_artifact_store import destroy_session_data
from app.dependencies import AppContainer, get_app_container

router = APIRouter()


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    container: AppContainer = Depends(get_app_container),
) -> dict[str, str | bool]:
    sid = (session_id or "").strip()
    if not sid:
        return {"ok": False, "session_id": "", "message": "session_id required"}
    destroy_session_data(sid, container.settings)
    return {"ok": True, "session_id": sid, "message": "session destroyed"}
