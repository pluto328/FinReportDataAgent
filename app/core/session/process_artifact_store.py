"""Persist intermediate data path catalog per session."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from app.config.settings import Settings, get_settings

CATALOG_FILENAME = "intermediate_data_catalog.json"
PROCESSED_SUBDIR = "processed"


def processed_dir(session_id: str, settings: Settings | None = None) -> Path:
    s = settings or get_settings()
    return s.cache_path / session_id / PROCESSED_SUBDIR


def catalog_path(session_id: str, settings: Settings | None = None) -> Path:
    s = settings or get_settings()
    return s.cache_path / session_id / CATALOG_FILENAME


def load_intermediate_catalog(session_id: str, settings: Settings | None = None) -> dict[str, str]:
    path = catalog_path(session_id, settings)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return {str(k): str(v) for k, v in raw.items()}
    except (json.JSONDecodeError, OSError, ValueError):
        pass
    return {}


def save_intermediate_catalog(
    session_id: str,
    catalog: dict[str, str],
    settings: Settings | None = None,
) -> None:
    s = settings or get_settings()
    path = catalog_path(session_id, s)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")


def merge_intermediate_catalog(
    session_id: str,
    updates: dict[str, str],
    settings: Settings | None = None,
) -> dict[str, str]:
    if not updates:
        return load_intermediate_catalog(session_id, settings)
    catalog = load_intermediate_catalog(session_id, settings)
    for path_key, desc in updates.items():
        if path_key and desc is not None:
            catalog[str(path_key)] = str(desc)
    save_intermediate_catalog(session_id, catalog, settings)
    return catalog


def format_intermediate_catalog(catalog: dict[str, str]) -> str:
    if not catalog:
        return "（暂无中间数据）"
    return "\n".join(f"- {path}: {desc}" for path, desc in catalog.items())


def clear_session_workspace(session_id: str, settings: Settings | None = None) -> None:
    """Clear processed files and intermediate catalog for a session."""
    s = settings or get_settings()
    proc = processed_dir(session_id, s)
    if proc.exists():
        shutil.rmtree(proc, ignore_errors=True)
    cat = catalog_path(session_id, s)
    if cat.exists():
        cat.unlink(missing_ok=True)
