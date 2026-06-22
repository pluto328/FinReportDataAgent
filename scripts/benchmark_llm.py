"""Benchmark LLM models: TTFT, total latency, JSON validity."""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config.settings import Settings
from app.infrastructure.llm_client import LLMClient

SAMPLE_PROMPT = """你是数据处理规划器。输出 JSON，不要其它内容：
{"steps":[{"tool":"pandas_execute","params":{"file_paths":["资产负债表.csv"],"code":"result=df.head(5)","artifact_name":"top5.csv","artifact_description":"前五"}}]}
用户需求:负债榜前五
已预览数据列:TOTAL_LIABILITIES,SECURITY_NAME_ABBR
"""

DEFAULT_MODELS: list[str] = []


async def _bench_one(client: LLMClient, *, runs: int = 3) -> dict:
    ttfts: list[float] = []
    totals: list[float] = []
    valid_json = 0
    for _ in range(runs):
        t0 = time.perf_counter()
        first: float | None = None
        parts: list[str] = []
        async for chunk in client.astream(SAMPLE_PROMPT):
            if first is None:
                first = time.perf_counter() - t0
            parts.append(chunk)
        total = time.perf_counter() - t0
        ttfts.append(first or total)
        totals.append(total)
        raw = "".join(parts)
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                json.loads(raw[start : end + 1])
                valid_json += 1
        except json.JSONDecodeError:
            pass
    return {
        "model": client.model,
        "runs": runs,
        "ttft_p50_ms": round(statistics.median(ttfts) * 1000, 1),
        "total_p50_ms": round(statistics.median(totals) * 1000, 1),
        "total_p95_ms": round(sorted(totals)[max(0, int(len(totals) * 0.95) - 1)] * 1000, 1),
        "json_valid_rate": valid_json / runs if runs else 0,
    }


async def main() -> None:
    settings = Settings()
    models = DEFAULT_MODELS.copy()
    for name in (
        settings.llm_model,
        settings.llm_model_planner,
        settings.llm_model_data,
        settings.llm_model_reporter,
    ):
        if name and name not in models:
            models.append(name)
    if not models:
        print("Set LLM_MODEL (and optional LLM_MODEL_PLANNER/DATA/REPORTER) in .env")
        return
    runs = 3
    print(f"Benchmark prompt length: {len(SAMPLE_PROMPT)} chars, runs={runs}\n")
    rows: list[dict] = []
    for name in dict.fromkeys(models):
        client = LLMClient(settings, model=name)
        row = await _bench_one(client, runs=runs)
        rows.append(row)
        print(
            f"{row['model']}: TTFT p50={row['ttft_p50_ms']}ms "
            f"total p50={row['total_p50_ms']}ms p95={row['total_p95_ms']}ms "
            f"json_valid={row['json_valid_rate']:.0%}"
        )
    out = _ROOT / "eval" / "llm_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    asyncio.run(main())
