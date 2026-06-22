"""retrieve_knowledge_node — 文本知识库混合检索。

功能：
  当 enable_knowledge_retrieve 时，用 text_query 调用 EnsembleRetriever。

输入（AgentState）：
  node_flags, text_query, user_query

输出（state patch）：
  knowledge_chunks, nodes_traversed
"""

from __future__ import annotations

import asyncio

from app.core.agent.nodes._debug_runtime import print_node_result, sample_state, stub_runtime
from app.core.agent.nodes._helpers import append_node
from app.core.agent.nodes._node_log import node_logger
from app.core.agent.state import AgentRuntime, AgentState, dedupe_scored_chunks


async def retrieve_knowledge_node(state: AgentState, runtime: AgentRuntime) -> dict:
    log = node_logger(runtime.settings, "retrieve_knowledge")
    flags = state.get("node_flags")
    if state.get("retrieval_from_reporter"):
        enabled = bool(state.get("supplemental_retrieve_knowledge"))
    else:
        enabled = bool(flags and flags.enable_knowledge_retrieve)
    if not enabled:
        log.start(enabled=False)
        log.info("知识检索未启用，跳过")
        log.end(skipped=True)
        return append_node(state, "retrieve_knowledge")
    query = state.get("text_query") or state.get("user_query", "")
    log.start(query=query, enabled=True)
    log.info("正在查询文档", query=query)
    hits = dedupe_scored_chunks(await runtime.text_retriever.search(query))
    min_score = runtime.settings.min_rerank_score
    log.info("文档检索完成", count=len(hits), min_rerank_score=min_score)
    log.end(count=len(hits), query=query)
    return {"knowledge_chunks": hits, **append_node(state, "retrieve_knowledge")}


async def debug_retrieve_knowledge_node() -> None:
    state = sample_state()
    runtime = stub_runtime()
    result = await retrieve_knowledge_node(state, runtime)
    print_node_result("retrieve_knowledge_node", result)


if __name__ == "__main__":
    asyncio.run(debug_retrieve_knowledge_node())
