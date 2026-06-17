"""Report-phase tools."""

from __future__ import annotations

from typing import Any

from app.core.tools.report.read_data_file import ReadDataFileTool

__all__ = ["ReadDataFileTool", "get_report_tools"]


def get_report_tools() -> list[Any]:
    return [ReadDataFileTool()]
