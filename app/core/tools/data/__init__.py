"""Data-processing tools."""

from __future__ import annotations

from typing import Any

from app.core.tools.data.data_filter import DataFilterTool
from app.core.tools.data.make_chart import MakeChartTool
from app.core.tools.data.pandas_execute import PandasExecuteTool
from app.core.tools.data.preview_read import PreviewReadTool
from app.core.tools.data.sql_execute import SqlExecuteTool

__all__ = [
    "DataFilterTool",
    "MakeChartTool",
    "PandasExecuteTool",
    "PreviewReadTool",
    "SqlExecuteTool",
    "get_data_tools",
]


def get_data_tools() -> list[Any]:
    return [
        PreviewReadTool(),
        DataFilterTool(),
        SqlExecuteTool(),
        PandasExecuteTool(),
        MakeChartTool(),
    ]
