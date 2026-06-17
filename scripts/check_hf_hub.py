"""Debug helper: verify Hugging Face Hub reachability and model metadata."""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "debug-72ff74.log"
SESSION = "72ff74"


def _log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # region agent log
    payload = {
        "sessionId": SESSION,
        "runId": "hf-hub-check",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    # endregion


def _tcp(host: str, port: int = 443, timeout: float = 5.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def main() -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from app.config.settings import get_settings
    from app.infrastructure.hf_hub_config import configure_hf_hub

    settings = get_settings()
    configure_hf_hub(settings)

    _log("H1", "check_hf_hub.py:tcp", "huggingface_co", {"ok": _tcp("huggingface.co")})
    endpoint = settings.hf_endpoint.rstrip("/")
    mirror_host = endpoint.replace("https://", "").replace("http://", "").split("/")[0]
    _log("H2", "check_hf_hub.py:tcp", "mirror_host", {"host": mirror_host, "ok": _tcp(mirror_host)})
    _log(
        "H3",
        "check_hf_hub.py:env",
        "hf_env",
        {
            "HF_ENDPOINT": os.environ.get("HF_ENDPOINT", ""),
            "HF_HOME": os.environ.get("HF_HOME", ""),
            "HF_HUB_DOWNLOAD_TIMEOUT": os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT", ""),
            "embed_model": settings.embed_model_name,
        },
    )

    hub_endpoint = ""
    try:
        from huggingface_hub.constants import ENDPOINT

        hub_endpoint = ENDPOINT
    except Exception as exc:
        hub_endpoint = f"import_error:{exc}"
    _log(
        "H3",
        "check_hf_hub.py:hub_constants",
        "hub_constants_endpoint",
        {"endpoint": str(hub_endpoint)},
    )

    meta_ok = False
    meta_err = ""
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            settings.embed_model_name,
            "config.json",
            local_files_only=False,
        )
        meta_ok = Path(path).is_file()
        meta_err = "" if meta_ok else "download returned no file"
    except Exception as exc:
        meta_err = str(exc)
    _log(
        "H4",
        "check_hf_hub.py:hub_download",
        "model_config_download",
        {"ok": meta_ok, "model": settings.embed_model_name, "error": meta_err[:300]},
    )

    if meta_ok:
        print(f"HF Hub OK via {endpoint} (hub ENDPOINT={hub_endpoint})")
        return 0
    print("HF Hub check failed. See debug-72ff74.log")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
