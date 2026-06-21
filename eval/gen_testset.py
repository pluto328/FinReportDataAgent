"""Generate evaluation queries from random indexed document chunks (LLM)."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.common.logger import logger, setup_logger
from app.config.settings import get_settings
from app.infrastructure.es_client import ESClient
from app.infrastructure.llm_client import LLMClient


async def _fetch_random_chunks(es: ESClient, index: str, n: int, seed: int) -> list[dict]:
    count_resp = await es._client.count(
        index=index,
        body={"query": {"bool": {"filter": [{"term": {"status": "online"}}]}}},
    )
    total = int(count_resp.get("count", 0))
    if total <= 0:
        return []

    fetch_n = min(total, max(n * 3, n))
    resp = await es._client.search(
        index=index,
        body={
            "size": fetch_n,
            "query": {"bool": {"filter": [{"term": {"status": "online"}}]}},
            "_source": ["chunk_id", "doc_id", "source_file", "content"],
        },
    )
    chunks: list[dict] = []
    for hit in resp["hits"]["hits"]:
        src = hit["_source"]
        chunk_id = str(src.get("chunk_id", hit.get("_id", "")))
        content = str(src.get("content", "") or "")
        if not chunk_id or not content.strip():
            continue
        chunks.append(
            {
                "chunk_id": chunk_id,
                "doc_id": str(src.get("doc_id", "")),
                "source_file": str(src.get("source_file", "")),
                "content": content,
            }
        )
    random.Random(seed).shuffle(chunks)
    return chunks[:n]


async def _generate_query(llm: LLMClient, chunk: dict) -> str:
    content = str(chunk["content"])[:2000]
    prompt = (
        "你是检索评测数据标注员。根据下列文档片段，生成 1 条中文检索问句，"
        "要求：用户会用该问句在知识库中检索到该片段；问句应自然、具体，"
        "不要照抄片段标题或首句；只输出问句本身，不要解释。\n\n"
        f"片段:\n{content}"
    )
    raw = await llm.ainvoke(prompt)
    return (raw or "").strip().strip('"').strip("'")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Sample indexed chunks and LLM-generate queries")
    parser.add_argument("--size", type=int, default=100, help="Number of chunks to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for ES sampling")
    parser.add_argument("--out", default="", help="Output JSON path (default: eval/test_result/testset_*.json)")
    args = parser.parse_args()

    settings = get_settings()
    settings.ensure_directories()
    setup_logger(log_dir=str(settings.log_dir), level=settings.log_level)

    es = ESClient(settings)
    llm = LLMClient(settings)
    samples: list[dict] = []

    try:
        selected = await _fetch_random_chunks(es, settings.es_index_name, args.size, args.seed)
        if len(selected) < args.size:
            logger.warning("only {} online chunks in index, requested {}", len(selected), args.size)

        for i, chunk in enumerate(selected, 1):
            try:
                query = await _generate_query(llm, chunk)
            except Exception as exc:
                logger.warning("LLM failed for chunk {}: {}", chunk["chunk_id"], exc)
                continue
            if not query:
                continue
            samples.append(
                {
                    "query": query,
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "source_file": chunk["source_file"],
                    "content_preview": chunk["content"][:300],
                }
            )
            if i % 10 == 0:
                logger.info("generated {}/{} queries", len(samples), args.size)
    finally:
        await es.close()

    out_dir = settings.eval_result_path
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_file = Path(args.out) if args.out else out_dir / f"testset_{ts}.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "random_indexed_chunks",
        "sample_size": args.size,
        "seed": args.seed,
        "count": len(samples),
        "samples": samples,
    }
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("wrote {} ({} samples)", out_file, len(samples))


if __name__ == "__main__":
    asyncio.run(main())
