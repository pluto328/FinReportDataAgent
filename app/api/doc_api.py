"""Document upload and sync API."""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, UploadFile

from app.core.ingestion.meta_extractor import STRUCTURED_EXTENSIONS
from app.core.ingestion.parser import TEXT_EXTENSIONS
from app.core.ingestion.updater import DocumentUpdater
from app.dependencies import AppContainer, get_app_container
from app.schemas.document import AssetKind
from app.schemas.query import UploadResponse

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_doc(
    file: UploadFile = File(...),
    container: AppContainer = Depends(get_app_container),
) -> UploadResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        target_dir = container.settings.raw_doc_path
        kind = AssetKind.TEXT
    elif suffix in STRUCTURED_EXTENSIONS:
        target_dir = container.settings.raw_structured_path
        kind = AssetKind.STRUCTURED
    else:
        target_dir = container.settings.raw_doc_path
        kind = AssetKind.TEXT

    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / (file.filename or f"upload_{uuid4().hex}{suffix}")
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    updater = DocumentUpdater(
        container.settings,
        container.es,
        container.vectors,
        container.embed,
    )
    await updater.sync_path(dest)
    return UploadResponse(doc_id=dest.stem, file_name=dest.name, asset_kind=kind.value)


@router.post("/sync")
async def sync_docs(container: AppContainer = Depends(get_app_container)) -> dict[str, str]:
    from app.core.ingestion.updater import sync_all

    await sync_all(
        container.settings,
        es=container.es,
        vectors=container.vectors,
        embed=container.embed,
    )
    return {"message": "sync completed"}
