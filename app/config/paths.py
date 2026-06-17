"""Cross-platform path normalization for configuration values."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

# app/config/paths.py → project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANAGED_PATH_FIELDS: tuple[str, ...] = (
    "log_dir",
    "raw_doc_path",
    "raw_structured_path",
    "cache_path",
    "vector_persist_path",
    "eval_result_path",
    "report_output_path",
    "hf_home",
)

_QUOTED_PATH = re.compile(r'^["\'](.+)["\']$')
_URL_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z\d+\-.]*://")


def _strip_env_path(raw: str) -> str:
    """Strip whitespace and optional quotes from .env path strings."""
    text = raw.strip()
    matched = _QUOTED_PATH.match(text)
    if matched:
        text = matched.group(1).strip()
    return text


def normalize_config_path(
    value: str | Path,
    *,
    base: Path = PROJECT_ROOT,
    field_name: str = "path",
    must_stay_under_base: bool = True,
) -> Path:
    """Resolve env/config paths consistently on Windows/Linux/macOS."""
    if isinstance(value, Path):
        raw = str(value)
    else:
        raw = _strip_env_path(value)

    if not raw:
        msg = f"{field_name} must not be empty"
        raise ValueError(msg)
    if _URL_SCHEME.match(raw):
        msg = f"{field_name} looks like a URL, not a filesystem path: {raw!r}"
        raise ValueError(msg)

    # pathlib handles mixed `/` and `\` on Windows; expanduser for `~`
    candidate = Path(raw).expanduser()

    if not candidate.is_absolute():
        # Drop redundant leading `./` segments before joining project root
        parts = candidate.parts
        while parts and parts[0] in {".", ""}:
            parts = parts[1:]
        candidate = base.joinpath(*parts) if parts else base

    # strict=False: path need not exist yet (ensure_directories creates later)
    resolved = candidate.resolve(strict=False)

    if must_stay_under_base:
        base_resolved = base.resolve(strict=False)
        try:
            resolved.relative_to(base_resolved)
        except ValueError as exc:
            msg = (
                f"{field_name} must stay under project root "
                f"{base_resolved}; got {resolved}"
            )
            raise ValueError(msg) from exc

    return resolved


def as_posix(path: Path) -> str:
    """Return forward-slash path for logs and cross-platform display."""
    return path.as_posix()


@dataclass(frozen=True, slots=True)
class PathManager:
    """Grouped, finalized paths for runtime use."""

    project_root: Path
    log_dir: Path
    raw_doc_path: Path
    raw_structured_path: Path
    cache_path: Path
    vector_persist_path: Path
    eval_result_path: Path
    report_output_path: Path
    hf_home: Path

    @classmethod
    def from_values(
        cls,
        *,
        project_root: Path = PROJECT_ROOT,
        log_dir: Path,
        raw_doc_path: Path,
        raw_structured_path: Path,
        cache_path: Path,
        vector_persist_path: Path,
        eval_result_path: Path,
        report_output_path: Path,
        hf_home: Path,
    ) -> PathManager:
        kwargs = {
            "log_dir": log_dir,
            "raw_doc_path": raw_doc_path,
            "raw_structured_path": raw_structured_path,
            "cache_path": cache_path,
            "vector_persist_path": vector_persist_path,
            "eval_result_path": eval_result_path,
            "report_output_path": report_output_path,
            "hf_home": hf_home,
        }
        finalized = {
            name: normalize_config_path(
                path,
                base=project_root,
                field_name=name,
                must_stay_under_base=True,
            )
            for name, path in kwargs.items()
        }
        return cls(project_root=project_root.resolve(strict=False), **finalized)

    def all_managed(self) -> tuple[Path, ...]:
        return (
            self.log_dir,
            self.raw_doc_path,
            self.raw_structured_path,
            self.cache_path,
            self.vector_persist_path,
            self.eval_result_path,
            self.report_output_path,
            self.hf_home,
        )

    def ensure_all(self) -> None:
        """Create managed directories; uses os.makedirs for cross-platform safety."""
        for path in self.all_managed():
            os.makedirs(path, exist_ok=True)

    def display_map(self) -> dict[str, str]:
        """Posix paths for logging without platform-specific separators."""
        return {name: as_posix(getattr(self, name)) for name in MANAGED_PATH_FIELDS}
