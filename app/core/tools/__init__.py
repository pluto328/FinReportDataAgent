"""Agent 工具注册框架 — plan / data / report 三大模块。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "BaseTool",
    "ToolRegistry",
    "get_data_registry",
    "get_default_registry",
    "get_plan_registry",
    "get_report_registry",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "BaseTool": ("app.core.tools.base_tool", "BaseTool"),
    "ToolRegistry": ("app.core.tools.registry", "ToolRegistry"),
    "get_default_registry": ("app.core.tools.registry", "get_default_registry"),
    "get_plan_registry": ("app.core.tools.registry", "get_plan_registry"),
    "get_data_registry": ("app.core.tools.registry", "get_data_registry"),
    "get_report_registry": ("app.core.tools.registry", "get_report_registry"),
}

if TYPE_CHECKING:
    from app.core.tools.base_tool import BaseTool
    from app.core.tools.registry import ToolRegistry, get_default_registry


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    module_path, attr = _EXPORTS[name]
    module = __import__(module_path, fromlist=[attr])
    return getattr(module, attr)
