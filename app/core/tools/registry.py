"""Tool registry."""

from __future__ import annotations

from typing import Any

from app.core.tools.base_tool import BaseTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_names(self) -> list[str]:
        return list(self._tools.keys())


_default_registry: ToolRegistry | None = None


def get_default_registry() -> ToolRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry()
    return _default_registry


def get_plan_registry() -> ToolRegistry:
    from app.core.tools.plan import get_plan_tools

    registry = ToolRegistry()
    for tool in get_plan_tools():
        registry.register(tool)
    return registry


def get_data_registry() -> ToolRegistry:
    from app.core.tools.data import get_data_tools

    registry = ToolRegistry()
    for tool in get_data_tools():
        registry.register(tool)
    return registry


def get_report_registry() -> ToolRegistry:
    from app.core.tools.report import get_report_tools

    registry = ToolRegistry()
    for tool in get_report_tools():
        registry.register(tool)
    return registry
