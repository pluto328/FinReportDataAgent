"""Structured data read/filter/aggregate helpers."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from app.config.settings import get_settings

_DANGEROUS_SQL = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|ATTACH|COPY|EXPORT|IMPORT|PRAGMA)\b",
    re.I,
)


def read_table(path: str, *, limit: int | None = None) -> pd.DataFrame:
    s = get_settings()
    unlimited = limit is not None and limit < 0
    max_rows = limit if (limit is not None and limit >= 0) else s.structured_query_max_rows
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p) if unlimited else pd.read_csv(p, nrows=max_rows)
    if suffix == ".tsv":
        kw = {"sep": "\t"}
        return pd.read_csv(p, **kw) if unlimited else pd.read_csv(p, nrows=max_rows, **kw)
    if suffix == ".xlsx":
        return pd.read_excel(p) if unlimited else pd.read_excel(p, nrows=max_rows)
    if suffix == ".xlsb":
        kw = {"engine": "pyxlsb"}
        return pd.read_excel(p, **kw) if unlimited else pd.read_excel(p, nrows=max_rows, **kw)
    if suffix == ".parquet":
        df = pd.read_parquet(p)
        return df if unlimited else df.head(max_rows)
    if suffix == ".feather":
        df = pd.read_feather(p)
        return df if unlimited else df.head(max_rows)
    if suffix == ".jsonl":
        return pd.read_json(p, lines=True) if unlimited else pd.read_json(p, lines=True, nrows=max_rows)
    raise ValueError(f"unsupported format: {suffix}")


def read_table_full(path: str) -> pd.DataFrame:
    return read_table(path, limit=-1)


def read_table_preview(path: str, *, rows: int = 20) -> pd.DataFrame:
    return read_table(path, limit=rows)


def prepare_dataframe_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame for tabular export (flatten MultiIndex, avoid Excel write errors)."""
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [_flatten_column_name(col) for col in out.columns]
    if isinstance(out.index, pd.MultiIndex):
        out = out.reset_index()
    seen: dict[str, int] = {}
    unique_cols: list[str] = []
    for col in out.columns:
        name = str(col)
        count = seen.get(name, 0)
        if count:
            unique_cols.append(f"{name}_{count + 1}")
        else:
            unique_cols.append(name)
        seen[name] = count + 1
    out.columns = unique_cols
    return out


def _flatten_column_name(col: object) -> str:
    if isinstance(col, tuple):
        parts = [str(c) for c in col if c is not None and str(c) != ""]
        return "_".join(parts) if parts else "column"
    return str(col)


def write_table(df: pd.DataFrame, path: Path) -> None:
    df = prepare_dataframe_for_export(df)
    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    if suffix == ".tsv":
        df.to_csv(path, sep="\t", index=False)
        return
    if suffix == ".xlsx":
        df.to_excel(path, index=False)
        return
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return
    if suffix == ".feather":
        df.to_feather(path)
        return
    if suffix == ".jsonl":
        df.to_json(path, orient="records", lines=True, force_ascii=False)
        return
    df.to_csv(path, index=False)


def filter_rows(df: pd.DataFrame, column: str, op: str, value: str) -> pd.DataFrame:
    if column not in df.columns:
        return df.head(0)
    series = df[column]
    if op == "eq":
        return df[series.astype(str) == value]
    if op == "contains":
        return df[series.astype(str).str.contains(value, na=False)]
    return df.head(0)


def aggregate_df(df: pd.DataFrame, group_by: str, agg_col: str, agg_fn: str = "sum") -> pd.DataFrame:
    if group_by not in df.columns or agg_col not in df.columns:
        return df.head(0)
    grouped = df.groupby(group_by)[agg_col]
    if agg_fn == "mean":
        return grouped.mean().reset_index()
    if agg_fn == "count":
        return grouped.count().reset_index()
    return grouped.sum().reset_index()


def _duckdb_read_expr(path: str) -> str:
    p = Path(path)
    suffix = p.suffix.lower()
    escaped = str(p).replace("\\", "/").replace("'", "''")
    if suffix == ".csv":
        return f"read_csv_auto('{escaped}')"
    if suffix == ".tsv":
        return f"read_csv_auto('{escaped}', delim='\\t')"
    if suffix in {".parquet", ".pq"}:
        return f"read_parquet('{escaped}')"
    if suffix == ".jsonl":
        return f"read_json_auto('{escaped}')"
    if suffix == ".xlsx":
        return f"read_excel('{escaped}')"
    raise ValueError(f"unsupported format for SQL: {suffix}")


def normalize_file_paths(file_path: str | list[str] | None = None, *, file_paths: list[str] | str | None = None) -> list[str]:
    """Normalize single or list file path kwargs into a non-empty path list."""
    raw: list[str] = []
    if file_paths is not None:
        if isinstance(file_paths, str):
            raw = [file_paths]
        else:
            raw = [str(p) for p in file_paths if p]
    elif file_path is not None:
        if isinstance(file_path, str):
            raw = [file_path] if file_path else []
        else:
            raw = [str(p) for p in file_path if p]
    return raw


def _sql_view_name(index: int) -> str:
    """第1张表 src，第2张 src1，第3张 src2 …"""
    return "src" if index == 0 else f"src{index}"


def _inject_sql_legacy_view_aliases(con, file_count: int) -> None:
    """兼容旧 prompt 中「第2个表 src2、第3个 src3」写法。"""
    if file_count == 2:
        con.execute("CREATE VIEW src2 AS SELECT * FROM src1")
    elif file_count >= 3:
        con.execute("CREATE VIEW src3 AS SELECT * FROM src2")


def execute_sql_on_files(file_paths: list[str], sql: str, *, limit: int | None = None) -> pd.DataFrame:
    """Run read-only SELECT against one or more structured files via DuckDB."""
    import duckdb

    if not file_paths:
        raise ValueError("file_paths is required")
    if _DANGEROUS_SQL.search(sql):
        raise ValueError("only SELECT queries are allowed")
    normalized = sql.strip().rstrip(";")
    if not re.match(r"^\s*SELECT\b", normalized, re.I):
        raise ValueError("SQL must start with SELECT")

    s = get_settings()
    max_rows = limit or s.structured_query_max_rows
    con = duckdb.connect(database=":memory:")
    try:
        for i, fp in enumerate(file_paths):
            view = _sql_view_name(i)
            read_expr = _duckdb_read_expr(fp)
            con.execute(f"CREATE VIEW {view} AS SELECT * FROM {read_expr}")
        _inject_sql_legacy_view_aliases(con, len(file_paths))
        wrapped = f"SELECT * FROM ({normalized}) AS q LIMIT {max_rows}"
        return con.execute(wrapped).df()
    finally:
        con.close()


def execute_sql_on_file(file_path: str, sql: str, *, limit: int | None = None) -> pd.DataFrame:
    """Run read-only SELECT against a structured file via DuckDB."""
    return execute_sql_on_files([file_path], sql, limit=limit)
