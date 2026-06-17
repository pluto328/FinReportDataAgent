"""Debug helper: check Elasticsearch connectivity and write NDJSON to debug log."""

from __future__ import annotations

import base64
import json
import socket
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "debug-72ff74.log"
SESSION = "72ff74"


def _log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # region agent log
    payload = {
        "sessionId": SESSION,
        "runId": "es-health-check",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    # endregion


def _auth_header(user: str, password: str) -> dict[str, str]:
    token = base64.b64encode(f"{user}:{password}".encode("ascii")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def main() -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    host = "127.0.0.1"
    port = 9200
    es_host = "http://127.0.0.1:9200"
    es_user = "elastic"
    es_password = "elastic123"

    # H1: port not listening (ES/Docker not running)
    sock_ok = False
    try:
        with socket.create_connection((host, port), timeout=3):
            sock_ok = True
    except OSError as exc:
        _log("H1", "check_es_health.py:socket", "tcp_connect_failed", {"error": str(exc), "host": host, "port": port})

    _log("H1", "check_es_health.py:socket", "tcp_connect_result", {"ok": sock_ok, "host": host, "port": port})
    if not sock_ok:
        print("ES not reachable (TCP). Is Docker / rag-es running?")
        return 1

    try:
        from app.config.settings import get_settings

        s = get_settings()
        es_host = s.es_host
        es_user = s.es_user
        es_password = s.es_password
        _log("H4", "check_es_health.py:settings", "env_es_host", {"es_host": s.es_host, "es_index": s.es_index_name})
    except Exception as exc:
        _log("H4", "check_es_health.py:settings", "settings_load_failed", {"error": str(exc)})

    url = es_host.rstrip("/")

    # H2: HTTP without auth — 401 means ES is up with security enabled
    no_auth_status: int | None = None
    req = Request(url, method="GET")
    try:
        with urlopen(req, timeout=5) as resp:
            body = resp.read(500).decode("utf-8", errors="replace")
            no_auth_status = resp.status
            _log("H2", "check_es_health.py:http", "http_no_auth_ok", {"status": resp.status, "body_preview": body[:200]})
    except HTTPError as exc:
        no_auth_status = exc.code
        if exc.code == 401:
            _log("H2", "check_es_health.py:http", "http_no_auth_401_es_up", {"status": 401, "url": url})
        else:
            _log("H2", "check_es_health.py:http", "http_no_auth_failed", {"error": str(exc), "status": exc.code, "url": url})
    except URLError as exc:
        _log("H2", "check_es_health.py:http", "http_no_auth_failed", {"error": str(exc.reason), "url": url})

    # H3: HTTP with Basic Auth header (avoid user:pass@ URL — breaks on Windows)
    req2 = Request(url, method="GET", headers=_auth_header(es_user, es_password))
    try:
        with urlopen(req2, timeout=5) as resp:
            body = resp.read(500).decode("utf-8", errors="replace")
            _log("H3", "check_es_health.py:http", "http_auth_ok", {"status": resp.status, "body_preview": body[:200]})
            print("ES OK:", body[:300])
            return 0
    except HTTPError as exc:
        body = exc.read(200).decode("utf-8", errors="replace") if exc.fp else ""
        _log(
            "H3",
            "check_es_health.py:http",
            "http_auth_http_error",
            {"status": exc.code, "body_preview": body[:200], "url": url},
        )
        if exc.code == 401:
            print("ES reachable but auth failed — check ES_USER / ES_PASSWORD in .env")
            return 1
    except URLError as exc:
        _log("H3", "check_es_health.py:http", "http_auth_failed", {"error": str(exc.reason), "url": url})

    print("ES not reachable. See debug-72ff74.log for details.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
