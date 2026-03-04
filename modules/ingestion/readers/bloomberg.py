from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from modules.ingestion.readers.base import BaseFileReader


def _format_date_as_period(value) -> Optional[str]:
    """Format a date value as YYYY/MM/DD for storage."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
        return value.strftime("%Y/%m/%d")

    raw = str(value).strip()
    if not raw:
        return None

    for fmt in ("%m/%d/%y", "%m/%d/%Y", "%Y-%m-%d", "%d/%m/%y", "%d/%m/%Y"):
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.strftime("%Y/%m/%d")
        except ValueError:
            continue
    return None


class BloombergFileReader(BaseFileReader):
    """Bloomberg Excel format with metadata extraction from header rows."""

    def __init__(self, data_header_row: int = 3) -> None:
        self.data_header_row = data_header_row

    def can_read(self, filepath: str) -> bool:
        return filepath.lower().endswith((".xlsx", ".xls"))

    def read(self, filepath: str) -> pd.DataFrame:
        return pd.read_excel(filepath, header=self.data_header_row)

    def extract_period(self, filepath: str) -> str:
        period = self._extract_period_from_cell_a2(filepath)
        return period if period else "UNKNOWN"

    def extract_index_code(self, filepath: str) -> Optional[str]:
        try:
            top = pd.read_excel(filepath, header=None, nrows=10)
        except Exception:
            return None

        for row_idx in range(top.shape[0] - 1):
            for col_idx in range(top.shape[1]):
                val = top.iat[row_idx, col_idx]
                if isinstance(val, str) and "universe" in val.lower() and "name" in val.lower():
                    below = top.iat[row_idx + 1, col_idx]
                    if below is not None and str(below).strip():
                        return str(below).strip()
        return None

    def _extract_period_from_cell_a2(self, filepath: str) -> Optional[str]:
        try:
            top = pd.read_excel(filepath, header=None, nrows=3)
            if top.shape[0] < 2 or top.shape[1] < 1:
                return None
            return _format_date_as_period(top.iat[1, 0])
        except Exception:
            return None
