"""Compile LangGraph agent — planner / retriever / process / reporter."""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, StateGraph

from app.core.agent.nodes import (
    chart_node,
    data_processor_node,
    data_tool_node,
    planner_node,
    planning_tool_node,
    process_executor_node,
    process_fanin_node,
    process_planner_node,
    process_worker_node,
    report_tool_node,
    reporter_node,
    retriever_node,
    route_after_chart_node,
    route_after_data_processor,
    route_after_data_tool,
    route_after_planner,
    route_after_planning_tool,
    route_after_process_executor,
    route_after_process_fanin,
    route_after_process_planner,
    route_after_report_tool,
    route_after_reporter,
    route_after_retriever,
    route_process_fanout,
)
from app.core.agent.state import AgentRuntime, AgentState


def build_agent_graph(runtime: AgentRuntime):
    graph = StateGraph(AgentState)

    graph.add_node("planner", partial(planner_node, runtime=runtime))
    graph.add_node("planning_tool", partial(planning_tool_node, runtime=runtime))
    graph.add_node("retriever", partial(retriever_node, runtime=runtime))
    graph.add_node("process_planner", partial(process_planner_node, runtime=runtime))
    graph.add_node("process_fanout", lambda state: state)
    graph.add_node("process_worker", partial(process_worker_node, runtime=runtime))
    graph.add_node("process_fanin", partial(process_fanin_node, runtime=runtime))
    graph.add_node("process_executor", partial(process_executor_node, runtime=runtime))
    graph.add_node("chart_node", partial(chart_node, runtime=runtime))
    graph.add_node("data_processor", partial(data_processor_node, runtime=runtime))
    graph.add_node("data_tool", partial(data_tool_node, runtime=runtime))
    graph.add_node("reporter", partial(reporter_node, runtime=runtime))
    graph.add_node("report_tool", partial(report_tool_node, runtime=runtime))

    graph.set_entry_point("planner")
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "planning_tool": "planning_tool",
            "planner_end": END,
            "retriever": "retriever",
            "process_planner": "process_planner",
            "data_processor": "data_processor",
            "reporter": "reporter",
        },
    )
    graph.add_conditional_edges(
        "planning_tool",
        route_after_planning_tool,
        {"planner": "planner"},
    )
    graph.add_conditional_edges(
        "retriever",
        route_after_retriever,
        {
            "process_planner": "process_planner",
            "data_processor": "data_processor",
            "reporter": "reporter",
        },
    )
    graph.add_conditional_edges(
        "process_planner",
        route_after_process_planner,
        {"process_fanout": "process_fanout", "reporter": "reporter"},
    )
    graph.add_conditional_edges(
        "process_fanout",
        route_process_fanout,
        ["process_worker", "process_executor"],
    )
    graph.add_edge("process_worker", "process_fanin")
    graph.add_conditional_edges(
        "process_fanin",
        route_after_process_fanin,
        {"chart_node": "chart_node", "reporter": "reporter"},
    )
    graph.add_conditional_edges(
        "process_executor",
        route_after_process_executor,
        {"chart_node": "chart_node", "reporter": "reporter"},
    )
    graph.add_conditional_edges(
        "chart_node",
        route_after_chart_node,
        {"reporter": "reporter"},
    )
    graph.add_conditional_edges(
        "data_processor",
        route_after_data_processor,
        {"data_tool": "data_tool", "reporter": "reporter", "data_processor": "data_processor"},
    )
    graph.add_conditional_edges(
        "data_tool",
        route_after_data_tool,
        {"data_processor": "data_processor", "reporter": "reporter"},
    )
    graph.add_conditional_edges(
        "reporter",
        route_after_reporter,
        {
            "retriever": "retriever",
            "report_tool": "report_tool",
            "reporter_end": END,
        },
    )
    graph.add_conditional_edges(
        "report_tool",
        route_after_report_tool,
        {"reporter": "reporter"},
    )

    return graph.compile()
