"""Application settings loaded from environment / .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config.paths import (
    MANAGED_PATH_FIELDS,
    PROJECT_ROOT,
    PathManager,
    normalize_config_path,
)

RetrieverName = Literal["es", "bm25", "dense"]
MetaRetrieverName = Literal["keyword", "dense"]


class Settings(BaseSettings):
    """Application settings loaded from .env."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- paths & runtime ---
    app_env: str = Field(default="dev", description="Runtime environment label")
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = Field(default="INFO")
    log_dir: Path = Field(default=Path("logs"))
    enable_agent_node_log: bool = Field(
        default=True,
        description="Emit structured start/end/info logs for each agent graph node",
    )
    agent_node_log_level: Literal["OFF", "ERROR", "INFO", "DEBUG"] = Field(
        default="INFO",
        description="Agent node log verbosity: OFF|ERROR|INFO|DEBUG (default full flow at INFO)",
    )
    sync_on_startup: bool = Field(default=True)
    sync_in_background: bool = Field(default=True)
    raw_doc_path: Path = Field(default=Path("data/raw_docs"))
    raw_structured_path: Path = Field(default=Path("data/raw_structured"))
    cache_path: Path = Field(default=Path("data/parsed_cache"))
    vector_persist_path: Path = Field(default=Path("data/persist_db"))
    eval_result_path: Path = Field(default=Path("eval/test_result"))
    report_output_path: Path = Field(default=Path("data/reports"))

    # --- LLM ---
    llm_base_url: str = Field(default="")
    llm_api_key: str = Field(default="")
    llm_model: str = Field(default="")
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)

    # --- embedding & rerank ---
    embed_model_name: str = Field(default="BAAI/bge-m3")
    rerank_model_name: str = Field(default="BAAI/bge-reranker-v2-m3")
    device: str = Field(default="cpu")
    hf_endpoint: str = Field(
        default="https://hf-mirror.com",
        description="Hugging Face Hub mirror (HF_ENDPOINT); use https://huggingface.co if direct access works",
    )
    hf_hub_download_timeout: int = Field(default=120, ge=10)
    hf_home: Path = Field(
        default=Path("data/hf_cache"),
        description="Local Hugging Face cache root (HF_HOME); keep on a drive with enough free space",
    )

    # --- chunking ---
    chunk_size: int = Field(default=512, ge=64)
    chunk_overlap: int = Field(default=64, ge=0)

    # --- elasticsearch ---
    es_host: str = Field(default="http://127.0.0.1:9200")
    es_user: str = Field(default="elastic")
    es_password: str = Field(default="elastic123")
    es_index_name: str = Field(default="rag_chunk_index")
    es_meta_index_name: str = Field(default="rag_meta_index")

    # --- chroma ---
    chroma_use_http: bool = Field(
        default=True,
        description="Use Chroma HTTP server (Docker/Linux); set false for local PersistentClient",
    )
    chroma_http_host: str = Field(default="127.0.0.1")
    chroma_http_port: int = Field(default=8001, ge=1, le=65535)
    chroma_collection: str = Field(default="rag_collection")
    chroma_meta_collection: str = Field(default="rag_meta_collection")

    # --- redis (optional) ---
    enable_redis: bool = Field(default=False)
    redis_url: str = Field(default="redis://127.0.0.1:6379/0")

    # --- cors ---
    cors_origins: str = Field(
        default="http://127.0.0.1:5500,http://localhost:5500",
    )

    # --- retrieval ablation ---
    enable_es: bool = Field(default=True)
    enable_bm25: bool = Field(default=True)
    enable_dense: bool = Field(default=True)
    dense_weight: float = Field(default=0.5, ge=0.0)
    es_weight: float = Field(default=0.3, ge=0.0)
    bm25_weight: float = Field(default=0.2, ge=0.0)
    base_top_k: int = Field(default=20, ge=1)
    final_top_k: int = Field(default=10, ge=1)
    min_retrieval_score: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Minimum bge-m3 cosine similarity to keep a retrieval hit",
    )

    # --- structured meta retrieval ---
    enable_meta_keyword: bool = Field(default=True)
    enable_meta_dense: bool = Field(default=True)
    meta_keyword_weight: float = Field(default=0.4, ge=0.0)
    meta_dense_weight: float = Field(default=0.6, ge=0.0)
    max_retrieval_rounds: int = Field(default=3, ge=1)
    max_plan_tool_steps: int = Field(default=3, ge=1)
    max_process_tool_steps: int = Field(default=5, ge=1)
    max_report_tool_steps: int = Field(default=5, ge=1)
    max_sql_retries: int = Field(default=3, ge=1)
    max_context_tool_steps: int = Field(default=5, ge=1)
    context_size_threshold_chars: int = Field(default=12000, ge=500)
    processed_data_preview_rows: int = Field(default=5, ge=1)
    structured_query_max_rows: int = Field(default=5000, ge=1)
    chart_task_timeout_sec: int = Field(default=120, ge=1)

    # finalized after validation; not loaded from env
    path_manager: PathManager | None = Field(default=None, exclude=True)

    @field_validator(*MANAGED_PATH_FIELDS, mode="before")
    @classmethod
    def _coerce_path(cls, value: str | Path, info: ValidationInfo) -> Path:
        field_name = info.field_name or "path"
        # First pass: normalize separators/quotes; strict root check in path_manager
        return normalize_config_path(
            value,
            base=PROJECT_ROOT,
            field_name=field_name,
            must_stay_under_base=False,
        )

    @field_validator("device")
    @classmethod
    def _normalize_device(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"cpu", "cuda", "mps"}:
            msg = f"unsupported DEVICE: {value!r}, expected cpu/cuda/mps"
            raise ValueError(msg)
        return normalized

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        return value.strip().upper()

    @model_validator(mode="after")
    def _validate_chunk_retrieval_and_paths(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            msg = "CHUNK_OVERLAP must be less than CHUNK_SIZE"
            raise ValueError(msg)
        if self.final_top_k > self.base_top_k:
            msg = "FINAL_TOP_K must not exceed BASE_TOP_K"
            raise ValueError(msg)
        if not (self.enable_es or self.enable_bm25 or self.enable_dense):
            msg = "at least one retriever must be enabled (ENABLE_ES/BM25/DENSE)"
            raise ValueError(msg)
        weight_sum = self.es_weight + self.bm25_weight + self.dense_weight
        if abs(weight_sum - 1.0) > 1e-6:
            msg = f"retrieval weights must sum to 1.0, got {weight_sum:.4f}"
            raise ValueError(msg)
        if not (self.enable_meta_keyword or self.enable_meta_dense):
            msg = "at least one meta retriever must be enabled"
            raise ValueError(msg)
        meta_sum = self.meta_keyword_weight + self.meta_dense_weight
        if abs(meta_sum - 1.0) > 1e-6:
            msg = f"meta retrieval weights must sum to 1.0, got {meta_sum:.4f}"
            raise ValueError(msg)

        object.__setattr__(
            self,
            "path_manager",
            PathManager.from_values(
                project_root=PROJECT_ROOT,
                log_dir=self.log_dir,
                raw_doc_path=self.raw_doc_path,
                raw_structured_path=self.raw_structured_path,
                cache_path=self.cache_path,
                vector_persist_path=self.vector_persist_path,
                eval_result_path=self.eval_result_path,
                report_output_path=self.report_output_path,
                hf_home=self.hf_home,
            ),
        )
        # Keep field values in sync with finalized absolute paths
        for name in MANAGED_PATH_FIELDS:
            object.__setattr__(self, name, getattr(self.path_manager, name))
        return self

    @property
    def project_root(self) -> Path:
        return self.path_manager.project_root if self.path_manager else PROJECT_ROOT

    @property
    def paths(self) -> PathManager:
        """Grouped path accessor; always available after model validation."""
        if self.path_manager is None:
            msg = "path_manager not initialized"
            raise RuntimeError(msg)
        return self.path_manager

    @property
    def is_dev(self) -> bool:
        return self.app_env.lower() == "dev"

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def enabled_retrievers(self) -> list[RetrieverName]:
        retrievers: list[RetrieverName] = []
        if self.enable_es:
            retrievers.append("es")
        if self.enable_bm25:
            retrievers.append("bm25")
        if self.enable_dense:
            retrievers.append("dense")
        return retrievers

    def retrieval_weights(self) -> dict[RetrieverName, float]:
        """Return weight map keyed by retriever name."""
        return {
            "es": self.es_weight,
            "bm25": self.bm25_weight,
            "dense": self.dense_weight,
        }

    def active_retrieval_weights(self) -> dict[RetrieverName, float]:
        """Weights for enabled retrievers only, re-normalized to sum 1.0."""
        weights = self.retrieval_weights()
        active = {name: weights[name] for name in self.enabled_retrievers}
        total = sum(active.values())
        if total <= 0:
            msg = "sum of enabled retriever weights must be positive"
            raise ValueError(msg)
        return {name: value / total for name, value in active.items()}

    @property
    def enabled_meta_retrievers(self) -> list[MetaRetrieverName]:
        retrievers: list[MetaRetrieverName] = []
        if self.enable_meta_keyword:
            retrievers.append("keyword")
        if self.enable_meta_dense:
            retrievers.append("dense")
        return retrievers

    def meta_retrieval_weights(self) -> dict[MetaRetrieverName, float]:
        return {"keyword": self.meta_keyword_weight, "dense": self.meta_dense_weight}

    def active_meta_retrieval_weights(self) -> dict[MetaRetrieverName, float]:
        weights = self.meta_retrieval_weights()
        active = {name: weights[name] for name in self.enabled_meta_retrievers}
        total = sum(active.values())
        if total <= 0:
            msg = "sum of enabled meta retriever weights must be positive"
            raise ValueError(msg)
        return {name: value / total for name, value in active.items()}

    def ensure_directories(self) -> None:
        """Create managed data/log/eval directories if missing."""
        self.paths.ensure_all()


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings singleton."""
    return Settings()
