"""Apply Hugging Face Hub environment before model downloads."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from app.config.settings import Settings

_LOG_PATH = Path(__file__).resolve().parents[2] / "debug-72ff74.log"
_SESSION = "72ff74"


def _agent_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # region agent log
    payload = {
        "sessionId": _SESSION,
        "runId": "hf-hub-config",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with _LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    # endregion


def configure_hf_hub(settings: Settings) -> None:
    """Set HF mirror, cache dir, and timeout env vars (before hub downloads)."""
    endpoint = (settings.hf_endpoint or "").strip().rstrip("/")
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(settings.hf_hub_download_timeout)

    hf_home = settings.hf_home.resolve()
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    cache_drive = Path(hf_home.drive or hf_home.anchor)
    free_gb = round(shutil.disk_usage(cache_drive).free / (1024**3), 2)

    hub_endpoint = "not_imported_yet"
    try:
        from huggingface_hub.constants import ENDPOINT as hub_endpoint  # noqa: PLC0415
    except Exception:
        pass

    torch_version = "not_imported_yet"
    try:
        import torch  # noqa: PLC0415

        torch_version = torch.__version__
    except Exception:
        pass

    _agent_log(
        "H5",
        "hf_hub_config.py:configure",
        "hf_env_applied",
        {
            "settings_endpoint": endpoint,
            "HF_ENDPOINT": os.environ.get("HF_ENDPOINT", ""),
            "HF_HOME": os.environ.get("HF_HOME", ""),
            "cache_drive_free_gb": free_gb,
            "hub_constants_endpoint": str(hub_endpoint),
            "torch_version": torch_version,
        },
    )
