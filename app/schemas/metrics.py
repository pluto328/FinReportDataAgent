"""RAG evaluation metrics storage."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvalMetrics(BaseModel):
    """Single ablation run metrics."""

    scheme_name: str
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    faithfulness: float = 0.0
    config: dict[str, str | float | bool] = Field(default_factory=dict)
