"""Debug data_processor LLM prompt in isolation."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from app.core.agent.prompts.data_processor_prompt import build_data_processor_prompt
from scripts.prompt_debug.common import (
    SAMPLES_DIR,
    build_runtime,
    default_output_path,
    load_state,
    run_prompt,
)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run data_processor prompt against LLM")
    parser.add_argument("--query", default="", help="Override user_query")
    parser.add_argument(
        "--state",
        default=str(SAMPLES_DIR / "data_processor_default.json"),
        help="JSON state file",
    )
    parser.add_argument("--out", default="", help="Output markdown path")
    args = parser.parse_args()

    overrides = {"user_query": args.query} if args.query else {}
    state = load_state(args.state if args.state else None, **overrides)
    runtime = build_runtime()
    prompt = build_data_processor_prompt(state, runtime)
    out = Path(args.out) if args.out else default_output_path("data_processor")
    await run_prompt(phase="data_processor", prompt=prompt, llm=runtime.llm, output_path=out)


if __name__ == "__main__":
    asyncio.run(main())
