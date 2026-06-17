"""Debug reporter LLM prompts in isolation."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from app.core.agent.prompts.reporter_prompt import (
    build_answer_prompt,
    build_reporter_decision_prompt,
)
from scripts.prompt_debug.common import (
    SAMPLES_DIR,
    build_runtime,
    default_output_path,
    load_state,
    run_prompt,
)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run reporter prompts against LLM")
    parser.add_argument(
        "--mode",
        choices=["decision", "answer", "both"],
        default="both",
        help="Which reporter prompt to run",
    )
    parser.add_argument("--query", default="", help="Override user_query")
    parser.add_argument(
        "--state",
        default=str(SAMPLES_DIR / "reporter_default.json"),
        help="JSON state file",
    )
    parser.add_argument("--out", default="", help="Output markdown base path (suffix added)")
    args = parser.parse_args()

    overrides = {"user_query": args.query} if args.query else {}
    state = load_state(args.state if args.state else None, **overrides)
    runtime = build_runtime()

    if args.mode in ("decision", "both"):
        prompt = build_reporter_decision_prompt(state, runtime)
        out = (
            Path(args.out)
            if args.out
            else default_output_path("reporter_decision")
        )
        await run_prompt(phase="reporter_decision", prompt=prompt, llm=runtime.llm, output_path=out)

    if args.mode in ("answer", "both"):
        prompt = build_answer_prompt(state, runtime)
        out = Path(args.out) if args.out else default_output_path("reporter_answer")
        if args.mode == "both" and not args.out:
            out = default_output_path("reporter_answer")
        await run_prompt(phase="reporter_answer", prompt=prompt, llm=runtime.llm, output_path=out)


if __name__ == "__main__":
    asyncio.run(main())
