"""retriever node — parallel text + structured metadata retrieval."""

from __future__ import annotations

import asyncio

from app.core.agent.events import emit_node_start
from app.core.agent.nodes._debug_runtime import print_node_result, sample_state, stub_runtime
from app.core.agent.nodes._node_log import node_logger
from app.core.agent.nodes.retrieve_data import retrieve_data_node
from app.core.agent.nodes.retrieve_knowledge import retrieve_knowledge_node
from app.core.agent.state import AgentRuntime, AgentState


async def retriever_node(state: AgentState, runtime: AgentRuntime) -> dict:
    log = node_logger(runtime.settings, "retriever")
    flags = state.get("node_flags")
    text_q = state.get("text_query") or state.get("user_query", "")
    data_q = state.get("data_query") or state.get("user_query", "")
    log.start(
        text_query=text_q,
        data_query=data_q,
        knowledge=bool(flags and flags.enable_knowledge_retrieve),
        data=bool(flags and flags.enable_data_retrieve),
    )
    log.info("并行执行知识检索与元数据检索")
    await emit_node_start("retriever")
    k_res, d_res = await asyncio.gather(
        retrieve_knowledge_node(state, runtime),
        retrieve_data_node(state, runtime),
    )
    merged: dict = {
        "nodes_traversed": ["retriever"],
        "need_more_retrieval": False,
        "retrieval_from_reporter": False,
        "supplemental_retrieve_knowledge": False,
        "supplemental_retrieve_data": False,
    }
    for part in (k_res, d_res):
        for key, val in part.items():
            if key == "nodes_traversed":
                merged["nodes_traversed"].extend(val)
            else:
                merged[key] = val
    k_count = len(merged.get("knowledge_chunks") or [])
    d_count = len(merged.get("data_file_paths") or [])
    log.info("检索汇总", knowledge_chunks=k_count, data_files=d_count)
    log.end(knowledge_chunks=k_count, data_files=d_count)
    return merged


async def debug_retriever_node() -> None:
    state = sample_state()
    runtime = stub_runtime()
    result = await retriever_node(state, runtime)
    print_node_result("retriever_node", result)


if __name__ == "__main__":
    asyncio.run(debug_retriever_node())
