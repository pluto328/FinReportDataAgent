"""Planning-phase tools."""

from __future__ import annotations

from typing import Any

from app.core.tools.plan.load_history_context import LoadHistoryContextTool

__all__ = [
    "LoadHistoryContextTool",
    "get_plan_tools",
]


def get_plan_tools() -> list[Any]:
    return [
        LoadHistoryContextTool(),
    ]
