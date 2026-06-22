"""Compile LangGraph agent — 7 nodes per README architecture."""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, StateGraph

from app.core.agent.nodes import (
    data_processor_node,
    data_tool_node,
    planner_node,
    planning_tool_node,
    report_tool_node,
    reporter_node,
    retriever_node,
    route_after_data_processor,
    route_after_data_tool,
    route_after_planner,
    route_after_planning_tool,
    route_after_report_tool,
    route_after_reporter,
    route_after_retriever,
)
from app.core.agent.state import AgentRuntime, AgentState


def build_agent_graph(runtime: AgentRuntime):
    graph = StateGraph(AgentState)

    graph.add_node("planner", partial(planner_node, runtime=runtime))
    graph.add_node("planning_tool", partial(planning_tool_node, runtime=runtime))
    graph.add_node("retriever", partial(retriever_node, runtime=runtime))
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
        {"data_processor": "data_processor", "reporter": "reporter"},
    )
    graph.add_conditional_edges(
        "data_processor",
        route_after_data_processor,
        {"data_tool": "data_tool", "reporter": "reporter", "data_processor": "data_processor"},
    )
    graph.add_conditional_edges(
        "data_tool",
        route_after_data_tool,
        {"data_processor": "data_processor"},
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
