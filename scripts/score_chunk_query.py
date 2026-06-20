"""Score query–text relevance: dense cosine (diagnostic) and rerank score (production threshold)."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Allow: python scripts/score_chunk_query.py  (from repo root)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config.settings import get_settings
from app.infrastructure.embedding_service import EmbeddingService


async def _score_one(embed: EmbeddingService, query: str, text: str, *, with_rerank: bool) -> dict:
    settings = get_settings()
    sim = await embed.similarity(query, text)
    out: dict = {
        "query": query,
        "text_preview": text[:200] + ("…" if len(text) > 200 else ""),
        "cosine_similarity": round(sim, 6),
        "min_rerank_score": settings.min_rerank_score,
    }
    rerank_score = await embed.rerank_score(query, text)
    out["rerank_score"] = round(rerank_score, 6)
    out["passes_threshold"] = rerank_score >= settings.min_rerank_score
    if not with_rerank:
        out["note"] = "rerank_score is always computed; threshold uses cross-encoder rerank"
    return out


chunk = """在2026年3月6日举行的十四届全国人大四次会议经济主题记者会上明确提出，
国家将进一步深化“人工智能+”行动，赋能千行百业，服务千家万户。其中最受瞩目的量化指标是：到“十五五”末，我国人工智能相关产业的规模将增长到10万亿元以上。这一目标的提出，标志着人工智能已从单一的技术领域突破，升级为国家经济增长的核心战略引擎。与“十四五”期间强调技术攻关和基础设施建设不同，“十五五”的聚焦点在于“规模增长”与“深度融合”。
10万亿的量级，意味着AI相关产业将在此轮经济周期中承担起拉动内需、优化供给侧结构、提升全要素生产率的重任，
成为中国式现代化进程中的关键增长极。"""
query = "人工智能相关产业的规模将增长到10万亿元以上。"

chunk = "金融产业"
query = "龙虎榜"


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score query–chunk relevance (vector cosine + rerank threshold)"
    )
    parser.add_argument("--query", default=query, help="Search query text")
    parser.add_argument("--chunk", default=chunk, help="Chunk text content")
    parser.add_argument("--chunk-file", default="", help="Read chunk text from file (UTF-8)")
    parser.add_argument(
        "--chunks-json",
        default="",
        help='JSON file with list of {"text": "..."} objects for batch scoring',
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="(default behavior) include rerank_score and passes_threshold in output",
    )
    parser.add_argument("--out", default="", help="Optional JSON output path")
    args = parser.parse_args()

    settings = get_settings()
    embed = EmbeddingService(settings)

    results: list[dict] = []
    if args.chunks_json:
        items = json.loads(Path(args.chunks_json).read_text(encoding="utf-8"))
        for i, item in enumerate(items):
            text = str(item.get("text", "") or item.get("content", ""))
            if not text:
                continue
            row = await _score_one(embed, args.query, text, with_rerank=True)
            row["index"] = i
            results.append(row)
    else:
        text = args.chunk
        if args.chunk_file:
            text = Path(args.chunk_file).read_text(encoding="utf-8")
        if not text.strip():
            raise SystemExit("Provide --chunk, --chunk-file, or --chunks-json")
        results.append(await _score_one(embed, args.query, text, with_rerank=True))

    for row in results:
        print(json.dumps(row, ensure_ascii=False, indent=2))
        print()

    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
