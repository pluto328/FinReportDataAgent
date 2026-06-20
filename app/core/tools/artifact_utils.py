"""Persist processed data artifacts with unified naming."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from app.config.settings import Settings, get_settings
from app.core.session.process_artifact_store import get_session_catalog, processed_dir
from app.schemas.structured import DataProcessMode, ProcessedDataRef

_UNSAFE_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize_artifact_name(name: str) -> str:
    text = Path(str(name or "").strip()).name.strip()
    text = _UNSAFE_FILENAME.sub("_", text)
    return text.strip(" .")


def get_session_artifact_filenames(session_id: str, settings: Settings | None = None) -> set[str]:
    catalog = get_session_catalog(session_id, settings)
    return {Path(path).name for path in catalog if path}


def resolve_named_artifact_path(
    session_id: str,
    artifact_name: str,
    settings: Settings | None = None,
    *,
    default_suffix: str = ".csv",
) -> Path:
    name = sanitize_artifact_name(artifact_name)
    if not name:
        raise ValueError("artifact_name is required")
    suffix = Path(name).suffix
    if not suffix:
        ds = default_suffix if default_suffix.startswith(".") else f".{default_suffix}"
        name = f"{name}{ds}"
    existing = get_session_artifact_filenames(session_id, settings)
    out_dir = processed_dir(session_id, settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate = name
    stem = Path(name).stem
    ext = Path(name).suffix
    idx = 2
    while candidate in existing:
        candidate = f"{stem}_{idx}{ext}"
        idx += 1
    return out_dir / candidate


def build_processed_filename(source_path: str, *, suffix_override: str = "") -> str:
    p = Path(source_path)
    suffix = suffix_override or p.suffix or ".csv"
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    return f"{p.stem}_processed{suffix}"


def resolve_processed_path(
    source_path: str,
    session_id: str,
    settings: Settings | None = None,
    *,
    suffix_override: str = "",
    artifact_name: str = "",
) -> Path:
    if artifact_name:
        return resolve_named_artifact_path(
            session_id,
            artifact_name,
            settings,
            default_suffix=suffix_override or Path(source_path).suffix or ".csv",
        )
    s = settings or get_settings()
    out_dir = processed_dir(session_id, s)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = build_processed_filename(source_path, suffix_override=suffix_override)
    candidate = out_dir / name
    if not candidate.exists():
        return candidate
    stem = Path(name).stem
    ext = Path(name).suffix
    idx = 2
    while True:
        alt = out_dir / f"{stem}_{idx}{ext}"
        if not alt.exists():
            return alt
        idx += 1


def preview_dataframe_rows(df: pd.DataFrame, rows: int = 3) -> list[dict[str, Any]]:
    """Return first N rows as records for tool responses and prompts."""
    if df.empty:
        return []
    return df.head(rows).to_dict(orient="records")


def save_dataframe_processed(
    df: pd.DataFrame,
    source_path: str,
    session_id: str,
    settings: Settings | None = None,
    *,
    suffix_override: str = "",
    mode: DataProcessMode = "tool",
    artifact_name: str = "",
) -> ProcessedDataRef:
    from app.core.tools.structured_ops import write_table

    artifact = resolve_processed_path(
        source_path,
        session_id,
        settings,
        suffix_override=suffix_override,
        artifact_name=artifact_name,
    )
    write_table(df, artifact)
    byte_size = artifact.stat().st_size if artifact.exists() else 0
    preview_rows = (settings or get_settings()).processed_data_preview_rows
    preview = json.dumps(df.head(preview_rows).to_dict(orient="records"), ensure_ascii=False)[:2000]
    return ProcessedDataRef(
        path=str(artifact.resolve()),
        preview=preview,
        mode=mode,
        row_count=len(df),
        byte_size=byte_size,
        source_file=source_path,
    )


def load_artifact_text(path: str, max_chars: int | None = None) -> str:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt", ".json", ".jsonl", ".md"}:
        text = p.read_text(encoding="utf-8")
    elif suffix == ".parquet":
        text = pd.read_parquet(p).to_csv(index=False)
    elif suffix in {".xlsx", ".xlsb"}:
        text = pd.read_excel(p).to_csv(index=False)
    elif suffix == ".feather":
        text = pd.read_feather(p).to_csv(index=False)
    elif suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        return str(p.resolve())
    else:
        text = p.read_text(encoding="utf-8", errors="replace")
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars]
    return text


def estimate_context_chars(
    *,
    knowledge_texts: list[str],
    meta_texts: list[str],
    processed: list[ProcessedDataRef],
    plan_context: dict[str, Any],
    process_result: dict[str, Any],
) -> int:
    total = 0
    total += sum(len(t) for t in knowledge_texts)
    total += sum(len(t) for t in meta_texts)
    total += sum(len(p.preview) + len(p.path) for p in processed)
    total += len(json.dumps(plan_context, ensure_ascii=False))
    total += len(json.dumps(process_result, ensure_ascii=False))
    return total
