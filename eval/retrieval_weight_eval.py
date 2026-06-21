"""Evaluate ES+dense fusion weights: gold-pair scores and recall@5 (eval-only)."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.common.logger import logger, setup_logger
from app.config.settings import get_settings
from app.infrastructure.embedding_service import EmbeddingService
from app.infrastructure.es_client import ESClient
from app.infrastructure.vector_client import VectorClient

from eval.retrieval_eval_utils import (
    dense_weights,
    fuse_channel_hits,
    gold_norm_in_batch,
    hits_from_dense_raw,
    hits_from_es_raw,
    load_latest_testset,
)

BASE_TOP_K = 10
FINAL_TOP_K = 5
REPORT_PATH = ROOT / "eval" / "评测检索权重.md"


async def _channel_hits_for_query(
    es: ESClient,
    vectors: VectorClient,
    embed: EmbeddingService,
    query: str,
    base_k: int,
) -> tuple[list[dict], list[dict]]:
    es_raw = await es.search_chunks(query, base_k)
    qv = await embed.embed_query(query)
    dense_raw = await vectors.query_chunks(qv, base_k)
    return hits_from_es_raw(es_raw), hits_from_dense_raw(dense_raw)


async def step_score_pairs(
    samples: list[dict],
    es: ESClient,
    vectors: VectorClient,
    embed: EmbeddingService,
) -> list[dict]:
    rows: list[dict] = []
    for i, sample in enumerate(samples, 1):
        query = str(sample.get("query", "")).strip()
        gold = str(sample.get("chunk_id", "")).strip()
        if not query or not gold:
            continue
        es_hits, dense_hits = await _channel_hits_for_query(es, vectors, embed, query, BASE_TOP_K)
        es_score = gold_norm_in_batch(es_hits, gold)
        dense_score = gold_norm_in_batch(dense_hits, gold)
        rows.append(
            {
                "query": query,
                "chunk_id": gold,
                "es_score": es_score,
                "dense_score": dense_score,
                "gold_in_es_topk": es_score > 0,
                "gold_in_dense_topk": dense_score > 0,
            }
        )
        if i % 20 == 0:
            logger.info("scored {}/{}", i, len(samples))
    return rows


def step_best_weight_by_fusion_avg(score_rows: list[dict]) -> list[dict]:
    summary: list[dict] = []
    for dw in dense_weights():
        ew = round(1.0 - dw, 1)
        fused = [dw * r["dense_score"] + ew * r["es_score"] for r in score_rows]
        avg = sum(fused) / len(fused) if fused else 0.0
        summary.append(
            {
                "dense_weight": dw,
                "es_weight": ew,
                "avg_fused_score": avg,
            }
        )
    best = max(summary, key=lambda x: x["avg_fused_score"]) if summary else {}
    for row in summary:
        row["is_best"] = row.get("dense_weight") == best.get("dense_weight")
    return summary


async def step_recall_by_weight(
    samples: list[dict],
    es: ESClient,
    vectors: VectorClient,
    embed: EmbeddingService,
) -> list[dict]:
    results: list[dict] = []
    total = len(samples)
    for dw in dense_weights():
        ew = round(1.0 - dw, 1)
        hits_count = 0
        for sample in samples:
            query = str(sample.get("query", "")).strip()
            gold = str(sample.get("chunk_id", "")).strip()
            if not query or not gold:
                continue
            es_hits, dense_hits = await _channel_hits_for_query(es, vectors, embed, query, BASE_TOP_K)
            fused = fuse_channel_hits(es_hits, dense_hits, dense_weight=dw)
            top_ids = [cid for cid, _ in fused[:FINAL_TOP_K]]
            if gold in top_ids:
                hits_count += 1
        recall = hits_count / total if total else 0.0
        results.append(
            {
                "dense_weight": dw,
                "es_weight": ew,
                "recall_at_5": recall,
                "hits": hits_count,
                "total": total,
            }
        )
        logger.info("recall dense_w={} -> {:.3f}", dw, recall)
    return results


def _write_report(
    *,
    testset_path: Path,
    score_rows: list[dict],
    fusion_summary: list[dict],
    recall_rows: list[dict],
    scores_json: Path,
) -> None:
    best = next((r for r in fusion_summary if r.get("is_best")), fusion_summary[0] if fusion_summary else {})
    best_recall = max(recall_rows, key=lambda x: x["recall_at_5"]) if recall_rows else {}

    lines = [
        "# 评测检索权重",
        "",
        f"生成时间: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## 实验设置",
        "",
        f"- 评测集: `{testset_path}`",
        f"- 样本数: {len(score_rows)}",
        f"- 分数明细 JSON: `{scores_json}`",
        f"- 召回: `base_top_k={BASE_TOP_K}`, 融合后取 top `{FINAL_TOP_K}`",
        "- 融合: ES / Dense 均在各自 top-k batch 内 min-max 归一化后加权（与线上一致）",
        "- 未使用 Reranker（纯双通道融合召回实验）",
        "",
        "## 1. query–黄金 chunk 通道归一化分融合 — 平均融合分",
        "",
        "对每个 query，在 ES / Dense 各自 top-k 结果中取黄金 chunk 的 min-max 归一化分（不在 top-k 记 0），"
        "再按 `dense_w × dense_score + es_w × es_score` 对所有 query 求平均。",
        "",
        "| 向量权重 | ES权重 | 平均融合分 | 最优 |",
        "| --- | --- | --- | --- |",
    ]
    for row in fusion_summary:
        mark = "✓" if row.get("is_best") else ""
        lines.append(
            f"| {row['dense_weight']:.1f} | {row['es_weight']:.1f} | "
            f"{row['avg_fused_score']:.6f} | {mark} |"
        )
    lines.extend(
        [
            "",
            f"**按平均融合分最优向量权重: {best.get('dense_weight', '—')} "
            f"(ES={best.get('es_weight', '—')}, avg={best.get('avg_fused_score', 0):.6f})**",
            "",
            "## 2. 检索召回率 (Recall@5)",
            "",
            "对每个 query 做双通道检索融合，top-5 含黄金 chunk_id 记为命中。",
            "",
            "| 向量权重 | ES权重 | 召回率 | 命中/总数 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in recall_rows:
        lines.append(
            f"| {row['dense_weight']:.1f} | {row['es_weight']:.1f} | "
            f"{row['recall_at_5']:.4f} | {row['hits']}/{row['total']} |"
        )
    lines.extend(
        [
            "",
            f"**Recall@5 最高向量权重: {best_recall.get('dense_weight', '—')} "
            f"(recall={best_recall.get('recall_at_5', 0):.4f})**",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote report {}", REPORT_PATH)


async def main() -> None:
    parser = argparse.ArgumentParser(description="ES+dense weight tuning evaluation")
    parser.add_argument("--testset", default="", help="Path to testset JSON (default: latest)")
    parser.add_argument("--skip-recall", action="store_true", help="Only run score fusion analysis")
    args = parser.parse_args()

    settings = get_settings()
    settings.ensure_directories()
    setup_logger(log_dir=str(settings.log_dir), level=settings.log_level)

    if args.testset:
        testset_path = Path(args.testset)
        data = json.loads(testset_path.read_text(encoding="utf-8"))
        samples = list(data.get("samples") or [])
    else:
        testset_path, samples = load_latest_testset(settings.eval_result_path)
    if not samples:
        logger.error("no testset samples; run: poetry run python eval/gen_testset.py")
        return

    es = ESClient(settings)
    vectors = VectorClient(settings)
    embed = EmbeddingService(settings)

    try:
        logger.info("step 2: channel normalized scores for gold chunks (n={})", len(samples))
        score_rows = await step_score_pairs(samples, es, vectors, embed)

        out_dir = settings.eval_result_path
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        scores_json = out_dir / f"weight_eval_scores_{ts}.json"
        scores_json.write_text(json.dumps(score_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("saved scores {}", scores_json)

        logger.info("step 3: fusion weight sweep by average score")
        fusion_summary = step_best_weight_by_fusion_avg(score_rows)

        recall_rows: list[dict] = []
        if not args.skip_recall:
            logger.info("step 4: recall@5 for each weight (this may take a while)")
            recall_rows = await step_recall_by_weight(samples, es, vectors, embed)

        _write_report(
            testset_path=testset_path,
            score_rows=score_rows,
            fusion_summary=fusion_summary,
            recall_rows=recall_rows,
            scores_json=scores_json,
        )
    finally:
        await es.close()
        vectors.close()


if __name__ == "__main__":
    asyncio.run(main())
