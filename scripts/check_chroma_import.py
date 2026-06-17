"""Debug helper: verify Chroma import without onnxruntime."""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "debug-72ff74.log"
SESSION = "72ff74"


def _log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # region agent log
    payload = {
        "sessionId": SESSION,
        "runId": "chroma-import-check",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    # endregion


def main() -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    onnx_ok = False
    onnx_err = ""
    try:
        import onnxruntime  # noqa: F401

        onnx_ok = True
    except Exception as exc:
        onnx_err = str(exc)
    _log("H1", "check_chroma_import.py:onnx", "onnxruntime_import", {"ok": onnx_ok, "error": onnx_err})

    chroma_ok = False
    chroma_err = ""
    telemetry_noise = ""
    try:
        from app.infrastructure.vector_client import VectorClient
        from app.config.settings import get_settings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            VectorClient(get_settings())
        telemetry_noise = " ".join(str(w.message) for w in caught)
        chroma_ok = True
    except Exception as exc:
        chroma_err = str(exc)
    _log("H2", "check_chroma_import.py:vector_client", "vector_client_init", {"ok": chroma_ok, "error": chroma_err})
    _log(
        "H3",
        "check_chroma_import.py:telemetry",
        "telemetry_warnings",
        {
            "has_capture_error": "capture() takes 1 positional argument" in telemetry_noise,
            "warning_count": len(telemetry_noise.split("capture")) - 1 if telemetry_noise else 0,
            "preview": telemetry_noise[:300],
        },
    )

    if chroma_ok:
        print("Chroma VectorClient OK (no onnxruntime default embedding required)")
        return 0
    print("Chroma init failed. See debug-72ff74.log")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
