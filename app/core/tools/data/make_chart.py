"""Chart drawing tool registered under data_registry."""

from __future__ import annotations

from typing import Any

from app.config.settings import Settings, get_settings
from app.core.tools.artifact_utils import resolve_processed_path
from app.core.tools.base_tool import BaseTool
from app.core.tools.data.chart_render import render_chart
from app.core.tools.structured_ops import read_table
from app.schemas.structured import ChartSpec, ChartType


class MakeChartTool(BaseTool):
    name = "make_chart"
    description = (
        "根据结构化数据绘制表格/折线图/柱状图，保存为 png 或 csv 文件。"
        "入参：file_path（绝对路径）、chart_type（table|line|bar）、x_axis、y_axis、title、"
        "artifact_name（必填，保存文件名含后缀，如 chart_top5.png，不与已有中间数据文件名重复）、"
        "artifact_description（必填，图表中文说明，如「负债榜前五名柱状图」）。"
        "返回：path（保存后的绝对路径）。"
    )

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        file_path = str(kwargs["file_path"])
        session_id = str(kwargs.get("session_id", "default"))
        settings: Settings = kwargs.get("settings") or get_settings()
        chart_type = ChartType(kwargs.get("chart_type", ChartType.TABLE.value))
        spec = ChartSpec(
            need_chart=True,
            chart_type=chart_type,
            x_axis=str(kwargs.get("x_axis", "")),
            y_axis=str(kwargs.get("y_axis", "")),
            title=str(kwargs.get("title", kwargs.get("description", "chart"))),
        )
        df = read_table(file_path)
        suffix = ".csv" if chart_type == ChartType.TABLE else ".png"
        artifact_name = str(kwargs.get("artifact_name", ""))
        out_path = resolve_processed_path(
            file_path, session_id, settings, suffix_override=suffix, artifact_name=artifact_name
        )
        path = await render_chart(df, spec, out_path.parent, output_path=out_path)
        return {"path": path}
