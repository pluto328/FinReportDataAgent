"""Diagnose Docker registry connectivity; append NDJSON to debug-72ff74.log."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "debug-72ff74.log"
SESSION = "72ff74"


def _log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # region agent log
    payload = {
        "sessionId": SESSION,
        "runId": "docker-registry-check",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    # endregion


def _tcp(host: str, port: int, timeout: float = 8.0) -> tuple[bool, str]:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, ""
    except OSError as exc:
        return False, str(exc)


def main() -> int:
    targets = [
        ("registry-1.docker.io", 443, "H1"),
        ("docker.elastic.co", 443, "H2"),
        ("auth.docker.io", 443, "H3"),
    ]
    for host, port, hid in targets:
        ok, err = _tcp(host, port)
        _log(hid, "check_docker_registry.py:tcp", "registry_tcp", {"host": host, "port": port, "ok": ok, "error": err})

    try:
        proc = subprocess.run(
            ["docker", "info", "--format", "{{json .}}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        info = json.loads(proc.stdout) if proc.stdout.strip() else {}
        _log(
            "H4",
            "check_docker_registry.py:docker_info",
            "docker_proxy_config",
            {
                "http_proxy": info.get("HttpProxy", ""),
                "https_proxy": info.get("HttpsProxy", ""),
                "no_proxy": info.get("NoProxy", ""),
            },
        )
    except (subprocess.SubprocessError, json.JSONDecodeError, FileNotFoundError) as exc:
        _log("H4", "check_docker_registry.py:docker_info", "docker_info_failed", {"error": str(exc)})

    try:
        with urlopen("https://docker.elastic.co/v2/", timeout=10) as resp:
            _log("H2", "check_docker_registry.py:http", "elastic_registry_http", {"status": resp.status})
    except OSError as exc:
        _log("H2", "check_docker_registry.py:http", "elastic_registry_http_failed", {"error": str(exc)})

    print(f"Diagnostics written to {LOG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
