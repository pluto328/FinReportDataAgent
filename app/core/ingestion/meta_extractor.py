"""Structured file metadata extraction (schema only)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from app.core.ingestion.chunker import Chunker
from app.schemas.structured import MetaRecord, StructuredFormat

STRUCTURED_EXTENSIONS = {
    ".csv": StructuredFormat.CSV,
    ".tsv": StructuredFormat.TSV,
    ".xlsx": StructuredFormat.XLSX,
    ".xlsb": StructuredFormat.XLSB,
    ".parquet": StructuredFormat.PARQUET,
    ".feather": StructuredFormat.FEATHER,
    ".jsonl": StructuredFormat.JSONL,
}

_TIME_HINT = re.compile(r"(time|date|year|month|day|created|updated)", re.I)


class MetadataExtractor:
    """Extract table metadata without loading full datasets."""

    def extract(self, path: Path) -> list[MetaRecord]:
        suffix = path.suffix.lower()
        if suffix not in STRUCTURED_EXTENSIONS:
            return []
        fmt = STRUCTURED_EXTENSIONS[suffix]
        md5 = Chunker.file_md5(path)
        asset_id = Chunker.new_doc_id(path)
        if fmt == StructuredFormat.XLSX:
            return self._xlsx(path, asset_id, md5, fmt)
        if fmt == StructuredFormat.XLSB:
            return self._xlsb(path, asset_id, md5, fmt)
        if fmt == StructuredFormat.PARQUET:
            return [self._parquet(path, asset_id, md5, fmt)]
        if fmt == StructuredFormat.FEATHER:
            return [self._feather(path, asset_id, md5, fmt)]
        if fmt == StructuredFormat.JSONL:
            return [self._jsonl(path, asset_id, md5, fmt)]
        return [self._csv_tsv(path, asset_id, md5, fmt)]

    def _build(
        self,
        path: Path,
        asset_id: str,
        md5: str,
        fmt: StructuredFormat,
        columns: list[str],
        *,
        table_name: str = "",
        sheet_name: str = "",
    ) -> MetaRecord:
        time_cols = [c for c in columns if _TIME_HINT.search(c)]
        categories = [path.parent.name] if path.parent.name else []
        keywords = list({path.stem, table_name, sheet_name, *columns[:20], *categories})
        search_text = " ".join(filter(None, [path.stem, table_name, sheet_name, *columns, *categories]))
        return MetaRecord(
            asset_id=asset_id if not sheet_name else f"{asset_id}_{sheet_name}",
            file_path=str(path.resolve()),
            file_name=path.name,
            format=fmt,
            table_name=table_name or path.stem,
            sheet_name=sheet_name,
            columns=columns,
            time_columns=time_cols,
            categories=categories,
            keywords=keywords,
            search_text=search_text,
            md5=md5,
        )

    def _csv_tsv(self, path: Path, asset_id: str, md5: str, fmt: StructuredFormat) -> MetaRecord:
        sep = "\t" if fmt == StructuredFormat.TSV else ","
        df = pd.read_csv(path, sep=sep, nrows=0)
        return self._build(path, asset_id, md5, fmt, list(df.columns))

    def _xlsx(self, path: Path, asset_id: str, md5: str, fmt: StructuredFormat) -> list[MetaRecord]:
        xls = pd.ExcelFile(path)
        records: list[MetaRecord] = []
        for sheet in xls.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet, nrows=0)
            records.append(
                self._build(path, asset_id, md5, fmt, list(df.columns), sheet_name=sheet, table_name=sheet)
            )
        return records

    def _xlsb(self, path: Path, asset_id: str, md5: str, fmt: StructuredFormat) -> list[MetaRecord]:
        import pyxlsb

        records: list[MetaRecord] = []
        with pyxlsb.open_workbook(path) as wb:
            for name in wb.sheets:
                df = pd.read_excel(path, sheet_name=name, engine="pyxlsb", nrows=0)
                records.append(
                    self._build(path, asset_id, md5, fmt, list(df.columns), sheet_name=name, table_name=name)
                )
        return records

    def _parquet(self, path: Path, asset_id: str, md5: str, fmt: StructuredFormat) -> MetaRecord:
        import pyarrow.parquet as pq

        schema = pq.read_schema(path)
        columns = schema.names
        return self._build(path, asset_id, md5, fmt, columns)

    def _feather(self, path: Path, asset_id: str, md5: str, fmt: StructuredFormat) -> MetaRecord:
        import pyarrow.feather as feather

        table = feather.read_table(path, columns=[])
        return self._build(path, asset_id, md5, fmt, table.schema.names)

    def _jsonl(self, path: Path, asset_id: str, md5: str, fmt: StructuredFormat) -> MetaRecord:
        columns: list[str] = []
        with path.open(encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        columns = list(obj.keys())
                        break
                except json.JSONDecodeError:
                    continue
        return self._build(path, asset_id, md5, fmt, columns)
