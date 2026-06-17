"""Matplotlib chart rendering helpers for make_chart tool."""

from __future__ import annotations

import asyncio
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from app.schemas.structured import ChartSpec, ChartType


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
        plt.figure(figsize=(8, 4))
        if spec.chart_type == ChartType.LINE:
            plt.plot(df[spec.x_axis], df[spec.y_axis])
        else:
            plt.bar(df[spec.x_axis].astype(str), df[spec.y_axis])
        plt.title(spec.title or "chart")
        plt.xlabel(spec.x_axis)
        plt.ylabel(spec.y_axis)
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()

    await asyncio.to_thread(_plot)
    return str(fig_path.resolve())
