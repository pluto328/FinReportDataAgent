"""FastAPI dependency injection and app singletons."""

from __future__ import annotations

from dataclasses import dataclass, field

from fastapi import Request

from app.config.settings import Settings, get_settings
from app.core.agent.graph import build_agent_graph
from app.core.agent.state import AgentRuntime
from app.core.retrieval.dense_retriever import DenseRetriever
from app.core.retrieval.ensemble import EnsembleRetriever
from app.core.retrieval.es_retriever import ESRetriever
from app.core.retrieval.meta_dense_retriever import MetaDenseRetriever
from app.core.retrieval.meta_ensemble import MetaEnsembleRetriever
from app.core.retrieval.meta_keyword_retriever import MetaKeywordRetriever
from app.core.retrieval.reranker import Reranker
from app.core.tools.registry import ToolRegistry, get_data_registry, get_plan_registry, get_report_registry
from app.infrastructure.embedding_service import EmbeddingService
from app.infrastructure.es_client import ESClient
from app.infrastructure.llm_client import LLMClient, build_role_llm_client
from app.infrastructure.vector_client import VectorClient


@dataclass
class AppContainer:
    settings: Settings
    es: ESClient
    vectors: VectorClient
    embed: EmbeddingService
    llm: LLMClient
    text_retriever: EnsembleRetriever
    meta_retriever: MetaEnsembleRetriever
    plan_registry: ToolRegistry
    data_registry: ToolRegistry
    report_registry: ToolRegistry
    agent_runtime: AgentRuntime
    agent_graph: object = field(default=None)

    @classmethod
    def create(cls, settings: Settings | None = None) -> AppContainer:
        s = settings or get_settings()
        es = ESClient(s)
        vectors = VectorClient(s)
        embed = EmbeddingService(s)
        llm = LLMClient(s)
        llm_planner = build_role_llm_client(s, "planner")
        llm_data = build_role_llm_client(s, "data")
        llm_reporter = build_role_llm_client(s, "reporter")
        es_r = ESRetriever(es)
        dense = DenseRetriever(vectors, embed)
        reranker = Reranker(embed)
        text_retriever = EnsembleRetriever(s, es_r, dense, reranker)
        meta_retriever = MetaEnsembleRetriever(
            s,
            MetaKeywordRetriever(es),
            MetaDenseRetriever(vectors, embed),
            reranker,
        )
        plan_registry = get_plan_registry()
        data_registry = get_data_registry()
        report_registry = get_report_registry()
        runtime = AgentRuntime(
            settings=s,
            llm=llm,
            llm_planner=llm_planner,
            llm_data=llm_data,
            llm_reporter=llm_reporter,
            text_retriever=text_retriever,
            meta_retriever=meta_retriever,
            plan_registry=plan_registry,
            data_registry=data_registry,
            report_registry=report_registry,
        )
        graph = build_agent_graph(runtime)
        return cls(
            settings=s,
            es=es,
            vectors=vectors,
            embed=embed,
            llm=llm,
            text_retriever=text_retriever,
            meta_retriever=meta_retriever,
            plan_registry=plan_registry,
            data_registry=data_registry,
            report_registry=report_registry,
            agent_runtime=runtime,
            agent_graph=graph,
        )

    async def shutdown(self) -> None:
        await self.es.close()
        self.vectors.close()


_container: AppContainer | None = None


def init_container(settings: Settings | None = None) -> AppContainer:
    global _container
    _container = AppContainer.create(settings)
    return _container


def get_container() -> AppContainer:
    if _container is None:
        return init_container()
    return _container


async def get_app_container(_request: Request) -> AppContainer:
    return get_container()
