"""Session lifecycle API."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

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


@router.get("/{session_id}/files/{filename}")
async def get_session_file(
    session_id: str,
    filename: str,
    container: AppContainer = Depends(get_app_container),
) -> FileResponse:
    sid = (session_id or "").strip()
    safe_name = Path(filename).name
    if not sid or not safe_name or safe_name != filename:
        raise HTTPException(status_code=400, detail="invalid session_id or filename")
    root = container.settings.cache_path / sid
    if not root.exists():
        raise HTTPException(status_code=404, detail="file not found")
    matches = [p for p in root.rglob(safe_name) if p.is_file() and p.name == safe_name]
    if not matches:
        raise HTTPException(status_code=404, detail="file not found")
    path = matches[0]
    media = "image/png"
    if path.suffix.lower() == ".csv":
        media = "text/csv"
    elif path.suffix.lower() in {".jpg", ".jpeg"}:
        media = "image/jpeg"
    elif path.suffix.lower() == ".gif":
        media = "image/gif"
    elif path.suffix.lower() == ".webp":
        media = "image/webp"
    return FileResponse(path, filename=safe_name, media_type=media)
