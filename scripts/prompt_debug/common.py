"""Shared helpers for prompt debug scripts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config.settings import get_settings
from app.core.agent.nodes._debug_runtime import sample_state
from app.core.agent.nodes._helpers import parse_llm_json
from app.core.agent.state import AgentRuntime, AgentState
from app.core.tools.registry import get_data_registry, get_plan_registry, get_report_registry
from app.infrastructure.llm_client import LLMClient, build_role_llm_client
from app.schemas.query import ChatMessage
from app.schemas.structured import NodeEnableFlags

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
OUTPUT_DIR = PROJECT_ROOT / "debug_output"


def _normalize_state_raw(raw: dict[str, Any]) -> dict[str, Any]:
    out = dict(raw)
    if "chat_history" in out:
        out["chat_history"] = [
            ChatMessage(**m) if isinstance(m, dict) else m for m in out["chat_history"]
        ]
    if isinstance(out.get("node_flags"), dict):
        out["node_flags"] = NodeEnableFlags(**out["node_flags"])
    return out


def load_state(path: str | None, **overrides: Any) -> AgentState:
    if path:
        raw = _normalize_state_raw(json.loads(Path(path).read_text(encoding="utf-8")))
        state = sample_state(**raw)
    else:
        state = sample_state()
    if overrides:
        state.update(overrides)  # type: ignore[arg-type]
    return state


def build_runtime() -> AgentRuntime:
    settings = get_settings()
    llm = LLMClient(settings)
    llm_planner = build_role_llm_client(settings, "planner")
    llm_data = build_role_llm_client(settings, "data")
    llm_reporter = build_role_llm_client(settings, "reporter")
    from app.core.retrieval.ensemble import EnsembleRetriever
    from app.core.retrieval.meta_ensemble import MetaEnsembleRetriever

    class _Noop:
        async def search(self, query: str, top_k: int | None = None) -> list:
            return []

    return AgentRuntime(
        settings=settings,
        llm=llm,
        llm_planner=llm_planner,
        llm_data=llm_data,
        llm_reporter=llm_reporter,
        text_retriever=_Noop(),  # type: ignore[arg-type]
        meta_retriever=_Noop(),  # type: ignore[arg-type]
        plan_registry=get_plan_registry(),
        data_registry=get_data_registry(),
        report_registry=get_report_registry(),
    )


async def run_prompt(
    *,
    phase: str,
    prompt: str,
    llm: LLMClient,
    output_path: Path | None = None,
) -> tuple[str, dict | None]:
    print(f"\n=== {phase} PROMPT ===\n")
    print(prompt)
    print(f"\n=== {phase} STREAMING OUTPUT ===\n")
    parts: list[str] = []
    async for chunk in llm.astream(prompt):
        print(chunk, end="", flush=True)
        parts.append(chunk)
    print("\n")
    raw = "".join(parts)
    parsed = None
    try:
        parsed = parse_llm_json(raw)
        print(f"=== {phase} PARSED JSON ===\n")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"(JSON parse failed: {exc})")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# Prompt Debug — {phase}",
            f"Time: {datetime.now().isoformat()}",
            "",
            "## Prompt",
            "",
            "```",
            prompt,
            "```",
            "",
            "## Raw Output",
            "",
            "```",
            raw,
            "```",
            "",
        ]
        if parsed is not None:
            lines.extend(
                [
                    "## Parsed JSON",
                    "",
                    "```json",
                    json.dumps(parsed, ensure_ascii=False, indent=2),
                    "```",
                    "",
                ]
            )
        output_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Saved: {output_path}")

    return raw, parsed


def default_output_path(phase: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUT_DIR / f"{ts}_{phase}.md"
