"""Agent module exports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["AgentState", "AgentRuntime", "build_agent_graph", "run_agent_stream", "sse_agent_events"]

_EXPORTS: dict[str, tuple[str, str]] = {
    "AgentState": ("app.core.agent.state", "AgentState"),
    "AgentRuntime": ("app.core.agent.state", "AgentRuntime"),
    "build_agent_graph": ("app.core.agent.graph", "build_agent_graph"),
    "run_agent_stream": ("app.core.agent.runner", "run_agent_stream"),
    "sse_agent_events": ("app.core.agent.runner", "sse_agent_events"),
}

if TYPE_CHECKING:
    from app.core.agent.graph import build_agent_graph
    from app.core.agent.runner import run_agent_stream, sse_agent_events
    from app.core.agent.state import AgentRuntime, AgentState


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    module_path, attr = _EXPORTS[name]
    module = __import__(module_path, fromlist=[attr])
    return getattr(module, attr)
