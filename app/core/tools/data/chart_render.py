"""Matplotlib chart rendering helpers for make_chart tool."""

from __future__ import annotations

import asyncio
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

from app.schemas.structured import ChartSpec, ChartType

_CJK_FONT_CONFIGURED = False


def _configure_cjk_font() -> None:
    """Pick a CJK-capable sans-serif font so axis labels render on Windows/Linux."""
    global _CJK_FONT_CONFIGURED
    if _CJK_FONT_CONFIGURED:
        return
    candidates = (
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
    )
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False
    _CJK_FONT_CONFIGURED = True


async def render_chart(
    df: pd.DataFrame,
    spec: ChartSpec,
    output_dir: Path,
    *,
    output_path: Path | None = None,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    if spec.chart_type == ChartType.TABLE:
        out = output_path or output_dir / "table_processed.csv"
        await asyncio.to_thread(df.to_csv, out, index=False)
        return str(out.resolve())

    if spec.x_axis not in df.columns or spec.y_axis not in df.columns:
        out = output_path or output_dir / "table_processed.csv"
        await asyncio.to_thread(df.to_csv, out, index=False)
        return str(out.resolve())

    fig_path = output_path or output_dir / "chart_processed.png"

    def _plot() -> None:
        _configure_cjk_font()
        fig, ax = plt.subplots(figsize=(8, 4))
        if spec.chart_type == ChartType.LINE:
            ax.plot(df[spec.x_axis], df[spec.y_axis])
        else:
            labels = df[spec.x_axis].astype(str)
            ax.bar(labels, df[spec.y_axis])
            if len(labels) > 6:
                ax.tick_params(axis="x", rotation=45)
                for label in ax.get_xticklabels():
                    label.set_ha("right")
        ax.set_title(spec.title or "chart")
        ax.set_xlabel(spec.x_axis)
        ax.set_ylabel(spec.y_axis)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)
        plt.close("all")

    await asyncio.to_thread(_plot)
    return str(fig_path.resolve())
