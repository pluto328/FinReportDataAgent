"""Debug reporter LLM prompts in isolation."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from app.core.agent.prompts.reporter_prompt import build_reporter_prompt
from scripts.prompt_debug.common import (
    SAMPLES_DIR,
    build_runtime,
    default_output_path,
    load_state,
    run_prompt,
)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run reporter prompt against LLM")
    parser.add_argument("--query", default="", help="Override user_query")
    parser.add_argument(
        "--state",
        default=str(SAMPLES_DIR / "reporter_default.json"),
        help="JSON state file",
    )
    parser.add_argument("--out", default="", help="Output markdown path")
    parser.add_argument(
        "--force-done",
        action="store_true",
        help="Simulate max tool steps reached (force action=done)",
    )
    args = parser.parse_args()

    overrides = {"user_query": args.query} if args.query else {}
    state = load_state(args.state if args.state else None, **overrides)
    runtime = build_runtime()

    prompt = build_reporter_prompt(state, runtime, force_done=args.force_done)
    out = Path(args.out) if args.out else default_output_path("reporter")
    await run_prompt(phase="reporter", prompt=prompt, llm=runtime.llm, output_path=out)


if __name__ == "__main__":
    asyncio.run(main())
