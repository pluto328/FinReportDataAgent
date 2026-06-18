"""Minimal runtime helpers for node-level debug scripts."""

from __future__ import annotations

import json
from typing import Any

from app.core.agent.state import AgentState
from app.core.tools.registry import get_data_registry, get_plan_registry, get_report_registry
from app.schemas.structured import NodeEnableFlags


def sample_state(**overrides: Any) -> AgentState:
    base: AgentState = {
        "user_query": "最近7天的销售数据汇总",
        "chat_history": [],
        "report_mode": False,
        "session_id": "debug-session",
        "node_flags": NodeEnableFlags(
            enable_knowledge_retrieve=True,
            enable_data_retrieve=True,
            enable_process=True,
        ),
        "text_query": "销售",
        "data_query": "销售 csv",
        "dataprocessplan": "读取销售数据并汇总",
        "plan_steps": [],
        "plan_step": 0,
        "plan_done": False,
        "pending_tool": None,
        "plan_context": {},
        "data_file_paths": [],
        "data_tool_steps": [],
        "process_step": 0,
        "process_done": False,
        "report_steps": [],
        "report_step": 0,
        "report_done": False,
        "report_context": {},
        "knowledge_chunks": [],
        "meta_hits": [],
        "nodes_traversed": [],
        "file_cache": [],
        "chart_artifacts": [],
    }
    base.update(overrides)
    return base


def print_node_result(node_name: str, result: dict[str, Any]) -> None:
    printable = {
        k: (v.model_dump() if hasattr(v, "model_dump") else v)
        for k, v in result.items()
    }
    print(f"=== {node_name} ===")
    print(json.dumps(printable, ensure_ascii=False, indent=2, default=str))


class _StubLLM:
    async def ainvoke(self, prompt: str) -> str:
        return (
            '{"action":"done","text_query":"stub","data_query":"stub",'
            '"dataprocessplan":"","enable_knowledge_retrieve":true,'
            '"enable_data_retrieve":false,"enable_process":false,'
            '"enable_chart":false,"enable_report":false}'
        )

    async def astream(self, prompt: str):
        text = await self.ainvoke(prompt)
        yield text


class _StubRetriever:
    async def search(self, query: str, top_k: int | None = None) -> list:
        return []


def stub_runtime() -> Any:
    from app.config.settings import get_settings
    from app.core.agent.state import AgentRuntime

    settings = get_settings()
    return AgentRuntime(
        settings=settings,
        llm=_StubLLM(),  # type: ignore[arg-type]
        text_retriever=_StubRetriever(),  # type: ignore[arg-type]
        meta_retriever=_StubRetriever(),  # type: ignore[arg-type]
        plan_registry=get_plan_registry(),
        data_registry=get_data_registry(),
        report_registry=get_report_registry(),
    )
