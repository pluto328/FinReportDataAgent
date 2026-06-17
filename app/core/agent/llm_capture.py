"""Optional capture of all LLM prompts and raw outputs during an agent run."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config.paths import PROJECT_ROOT

_records: list[dict[str, Any]] = []
_session_query: str = ""
_output_path: Path | None = None


def is_enabled() -> bool:
    return os.getenv("CAPTURE_LLM_IO", "").lower() in ("1", "true", "yes")


def start_capture(query: str, *, output_dir: Path | None = None) -> Path | None:
    global _records, _session_query, _output_path
    if not is_enabled():
        return None
    _records = []
    _session_query = query
    out_dir = output_dir or (PROJECT_ROOT / "debug_output" / "llm_io")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() else "_" for c in query[:30])
    _output_path = out_dir / f"{ts}_{safe}.md"
    return _output_path


def record_llm_call(*, phase: str, purpose: str, prompt: str, output: str) -> None:
    if not is_enabled():
        return
    entry = {
        "index": len(_records) + 1,
        "phase": phase,
        "purpose": purpose,
        "prompt": prompt,
        "output": output,
        "time": datetime.now().isoformat(),
    }
    _records.append(entry)
    _flush_markdown()


def record_llm_invoke(*, phase: str, purpose: str, prompt: str, output: str) -> None:
    record_llm_call(phase=phase, purpose=purpose, prompt=prompt, output=output)


def _flush_markdown() -> None:
    if _output_path is None:
        return
    lines = [
        "# LLM Prompt / Output Capture",
        "",
        f"Query: {_session_query}",
        f"Captured calls: {len(_records)}",
        "",
    ]
    for r in _records:
        lines.extend(
            [
                f"## {r['index']}. {r['phase']} ({r['purpose']})",
                "",
                f"Time: {r['time']}",
                "",
                "### Prompt",
                "",
                "```",
                r["prompt"],
                "```",
                "",
                "### Output",
                "",
                "```",
                r["output"],
                "```",
                "",
            ]
        )
    _output_path.write_text("\n".join(lines), encoding="utf-8")


def finalize_capture() -> Path | None:
    _flush_markdown()
    if _output_path and _output_path.exists():
        json_path = _output_path.with_suffix(".json")
        json_path.write_text(
            json.dumps(
                {"query": _session_query, "records": _records},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return _output_path
    return None


def get_records() -> list[dict[str, Any]]:
    return list(_records)
