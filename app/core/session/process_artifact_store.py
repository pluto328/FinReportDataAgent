"""Persist intermediate data path catalog per session."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from app.config.settings import Settings, get_settings

CATALOG_FILENAME = "intermediate_data_catalog.json"
PROCESSED_SUBDIR = "processed"

# session_id -> {absolute_path: description}
_session_catalogs: dict[str, dict[str, str]] = {}


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
    """Backward-compatible alias; prefer register_session_artifacts."""
    return register_session_artifacts(session_id, updates, settings)


def get_session_catalog(session_id: str, settings: Settings | None = None) -> dict[str, str]:
    """Return session-level path catalog (memory, loaded from disk on first access)."""
    if not session_id:
        return {}
    if session_id not in _session_catalogs:
        _session_catalogs[session_id] = load_intermediate_catalog(session_id, settings)
    return dict(_session_catalogs[session_id])


def register_session_artifacts(
    session_id: str,
    updates: dict[str, str],
    settings: Settings | None = None,
) -> dict[str, str]:
    """Register tool output paths and descriptions into the session catalog."""
    if not session_id:
        return {}
    if not updates:
        return get_session_catalog(session_id, settings)
    catalog = get_session_catalog(session_id, settings)
    for path_key, desc in updates.items():
        if path_key:
            catalog[str(path_key)] = str(desc) if desc is not None else ""
    _session_catalogs[session_id] = catalog
    save_intermediate_catalog(session_id, catalog, settings)
    return catalog


def reset_session_workspace(session_id: str, settings: Settings | None = None) -> None:
    """Clear in-memory catalog, processed files, and catalog file for a new session."""
    _session_catalogs.pop(session_id, None)
    clear_session_workspace(session_id, settings)


def format_intermediate_catalog(catalog: dict[str, str]) -> str:
    if not catalog:
        return "（暂无中间数据）"
    return "\n".join(f"- {path}: {desc}" for path, desc in catalog.items())


def clear_session_workspace(session_id: str, settings: Settings | None = None) -> None:
    """Clear processed files and intermediate catalog file on disk."""
    s = settings or get_settings()
    proc = processed_dir(session_id, s)
    if proc.exists():
        shutil.rmtree(proc, ignore_errors=True)
    cat = catalog_path(session_id, s)
    if cat.exists():
        cat.unlink(missing_ok=True)
