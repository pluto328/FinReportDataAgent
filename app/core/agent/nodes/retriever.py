"""retriever node — parallel text + structured metadata retrieval."""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.core.agent.events import emit_node_start, emit_progress_line, emit_progress_waiting
from app.core.agent.nodes._debug_runtime import print_node_result, sample_state, stub_runtime
from app.core.agent.nodes._helpers import load_file_previews_for_paths
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
    if state.get("retrieval_from_reporter"):
        enable_knowledge = bool(state.get("supplemental_retrieve_knowledge"))
        enable_data = bool(state.get("supplemental_retrieve_data"))
    else:
        enable_knowledge = bool(flags and flags.enable_knowledge_retrieve)
        enable_data = bool(flags and flags.enable_data_retrieve)
    wait_parts: list[str] = []
    if enable_knowledge and text_q.strip():
        wait_parts.append(f"正在检索 {text_q.strip()} 相关文本")
    if enable_data and data_q.strip():
        wait_parts.append(f"正在检索 {data_q.strip()} 相关数据")
    if wait_parts:
        await emit_progress_waiting("；".join(wait_parts), active=True)
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
    await emit_progress_waiting(active=False)
    retrieved_names: set[str] = set()
    for chunk in merged.get("knowledge_chunks") or []:
        src = chunk.chunk.source_file
        if src:
            retrieved_names.add(Path(src).name)
    for path in merged.get("data_file_paths") or []:
        if path:
            retrieved_names.add(Path(path).name)
    for hit in merged.get("meta_hits") or []:
        name = hit.record.file_name or hit.record.file_path
        if name:
            retrieved_names.add(Path(name).name)
    for name in sorted(retrieved_names):
        await emit_progress_line(f"检索到：{name}")
    file_paths = merged.get("data_file_paths") or []
    if file_paths and enable_data:
        paths = [str(p) for p in file_paths if p]
        previews = await asyncio.to_thread(
            load_file_previews_for_paths, paths, state.get("file_previews")
        )
        merged["file_previews"] = previews
        log.info("自动加载数据预览", files=len(previews))
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
