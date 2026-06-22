"""retrieve_data_node — 结构化元数据双路检索。

功能：
  当 enable_data_retrieve 时，用 data_query 调用 MetaEnsembleRetriever，提取 file_path。

输入（AgentState）：
  node_flags, data_query, user_query

输出（state patch）：
  meta_hits, data_file_paths, nodes_traversed
"""

from __future__ import annotations

import asyncio

from app.core.agent.nodes._debug_runtime import print_node_result, sample_state, stub_runtime
from app.core.agent.nodes._helpers import append_node
from app.core.agent.nodes._node_log import node_logger
from app.core.agent.state import AgentRuntime, AgentState, dedupe_scored_meta


async def retrieve_data_node(state: AgentState, runtime: AgentRuntime) -> dict:
    log = node_logger(runtime.settings, "retrieve_data")
    flags = state.get("node_flags")
    if state.get("retrieval_from_reporter"):
        enabled = bool(state.get("supplemental_retrieve_data"))
    else:
        enabled = bool(flags and flags.enable_data_retrieve)
    if not enabled:
        log.start(enabled=False)
        log.info("结构化数据检索未启用，跳过")
        log.end(skipped=True)
        return append_node(state, "retrieve_data")
    query = state.get("data_query") or state.get("user_query", "")
    log.start(query=query, enabled=True)
    log.info("正在检索结构化元数据", query=query)
    hits = dedupe_scored_meta(await runtime.meta_retriever.search(query))
    paths = [h.record.file_path for h in hits if h.record.file_path]
    log.info("元数据检索完成", hit_count=len(hits), file_paths=paths)
    log.end(hit_count=len(hits), paths=paths)
    return {
        "meta_hits": hits,
        "data_file_paths": paths,
        **append_node(state, "retrieve_data"),
    }


async def debug_retrieve_data_node() -> None:
    state = sample_state()
    runtime = stub_runtime()
    result = await retrieve_data_node(state, runtime)
    print_node_result("retrieve_data_node", result)


if __name__ == "__main__":
    asyncio.run(debug_retrieve_data_node())
