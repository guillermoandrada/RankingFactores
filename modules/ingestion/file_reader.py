"""
File reader for Excel and CSV financial data files.
"""

import os
import re
from datetime import datetime
from typing import Optional

import pandas as pd


def _date_to_period(value) -> Optional[str]:
    """
    Convert a date (mm/dd/yy or datetime) to period string 'YYYY Qn'.
    Returns None if parsing fails.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if hasattr(value, "year") and hasattr(value, "month"):
        # Already datetime-like (Excel often returns this)
        year, month = value.year, value.month
        quarter = (month - 1) // 3 + 1
        return f"{year} Q{quarter}"
    s = str(value).strip()
    if not s:
        return None
    for fmt in ("%m/%d/%y", "%m/%d/%Y", "%Y-%m-%d", "%d/%m/%y", "%d/%m/%Y"):
        try:
            dt = datetime.strptime(s, fmt)
            quarter = (dt.month - 1) // 3 + 1
            return f"{dt.year} Q{quarter}"
        except ValueError:
            continue
    return None


class FileReader:
    """
    Reads financial data from Excel or CSV files.
    Extracts metadata (period, index code) from file structure.
    """

    def __init__(self, data_header_row: int = 3) -> None:
        """
        Args:
            data_header_row: Zero-based row index where column headers are located.
            """
        self.data_header_row = data_header_row

    def read(self, filepath: str) -> pd.DataFrame:
        """
        Read the main data table from file.
        Tries Excel first, then CSV.
        """
        try:
            return pd.read_excel(filepath, header=self.data_header_row)
        except Exception as e_excel:
            try:
                return pd.read_csv(filepath, header=self.data_header_row)
            except Exception:
                raise ValueError(
                    f"Could not read file as Excel ({e_excel}) or CSV."
                ) from e_excel

    def extract_period_from_cell_a2(self, filepath: str) -> Optional[str]:
        """
        Read cell A2 from Excel (date in mm/dd/yy format) and infer quarter.
        Returns e.g. '2023 Q3' or None if not found/invalid.
        """
        try:
            top = pd.read_excel(filepath, header=None, nrows=3)
            if top.shape[0] < 2 or top.shape[1] < 1:
                return None
            val = top.iat[1, 0]  # A2 = row 1, col 0
            return _date_to_period(val)
        except Exception:
            return None

    def extract_period_from_filename(self, filepath: str) -> str:
        """Extract period (e.g. '2024 Q4') from filename. Fallback when A2 is unavailable."""
        base = os.path.basename(filepath)
        match = re.search(r"(\d{4}\s*Q[1-4])", base, re.IGNORECASE)
        return match.group(1).upper() if match else "UNKNOWN"

    def extract_period(self, filepath: str) -> str:
        """
        Infer period from cell A2 (mm/dd/yy) if available, else from filename.
        """
        period = self.extract_period_from_cell_a2(filepath)
        return period if period else self.extract_period_from_filename(filepath)

    def extract_index_code(self, filepath: str) -> Optional[str]:
        """
        Read top rows of Excel and find 'Universe Name' cell.
        Return the value in the cell below (e.g. 'B500').
        """
        try:
            top = pd.read_excel(filepath, header=None, nrows=10)
        except Exception:
            return None

        for row_idx in range(top.shape[0] - 1):
            for col_idx in range(top.shape[1]):
                val = top.iat[row_idx, col_idx]
                if (
                    isinstance(val, str)
                    and "universe" in val.lower()
                    and "name" in val.lower()
                ):
                    below = top.iat[row_idx + 1, col_idx]
                    if below is not None and str(below).strip():
                        return str(below).strip()
        return None
