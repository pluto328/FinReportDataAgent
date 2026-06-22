"""Run one agent query with LLM prompt/output capture enabled."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CAPTURE_LLM_IO"] = "1"

QUERY = "负债榜前五是哪些"


async def main() -> None:
    from app.core.agent.llm_capture import finalize_capture, get_records, start_capture
    from app.core.agent.runner import run_agent_stream
    from app.core.agent.events import ProgressEmitter
    from app.dependencies import init_container
    from app.schemas.query import SearchRequest

    session_id = uuid.uuid4().hex[:16]
    out = start_capture(QUERY)
    print(f"Capture file: {out}")
    print(f"Query: {QUERY}")
    print(f"Session: {session_id}")

    container = init_container()
    request = SearchRequest(
        query=QUERY,
        session_id=session_id,
        new_session=True,
        report_mode=False,
    )
    queue: asyncio.Queue = asyncio.Queue()
    emitter = ProgressEmitter(queue)

    try:
        response = await run_agent_stream(container, request, emitter)
        print("\n=== Agent done ===")
        print(f"status: {response.status}")
        print(f"answer: {response.answer[:500] if response.answer else ''}")
    finally:
        await container.shutdown()

    path = finalize_capture()
    records = get_records()
    print(f"\nSaved {len(records)} LLM calls -> {path}")
    if path and path.with_suffix(".json").exists():
        print(f"JSON: {path.with_suffix('.json')}")


if __name__ == "__main__":
    asyncio.run(main())
