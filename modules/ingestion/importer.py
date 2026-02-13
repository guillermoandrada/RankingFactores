"""
Orchestrates reading, validation, and persistence of financial data.
"""

import os
import sys
from typing import Optional

import pandas as pd

from modules.config import FIXED_COLUMNS, DEFAULT_INPUT_FILE
from modules.db import FinancialDatabase
from modules.ingestion.file_reader import FileReader
from modules.models import ImportResult


class DataImporter:
    """
    Orchestrates the full import pipeline:
    read file -> validate -> persist to database.
    """

    def __init__(
        self,
        file_reader: Optional[FileReader] = None,
        database: Optional[FinancialDatabase] = None,
    ) -> None:
        self._reader = file_reader or FileReader()
        self._db = database or FinancialDatabase()

    def import_file(self, filepath: str, verbose: bool = True) -> ImportResult:
        """
        Import a single file into the database.
        Returns ImportResult with counts.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        index_code = self._reader.extract_index_code(filepath)
        if verbose:
            print(f"--> Index code detected: {index_code}")

        df = self._reader.read(filepath)
        if verbose:
            print(f"Read file: {filepath}")

        self._validate_columns(df)

        df = df.dropna(subset=["Ticker"])
        period = self._reader.extract_period(filepath)
        if verbose:
            print(f"--> Period detected: {period}")

        result = self._db.save_fundamentals(df, period, index_code)

        if verbose:
            print(
                f"Success! Imported {result.companies_count} companies, "
                f"{result.metrics_count} metrics, {result.records_count} records."
            )
        return result

    def _validate_columns(self, df: pd.DataFrame) -> None:
        if "Ticker" not in df.columns:
            raise ValueError(
                f"Missing 'Ticker' column. Found: {list(df.columns)}"
            )
        missing = [c for c in FIXED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. Found: {list(df.columns)}"
            )


def main() -> None:
    """CLI entry point for data import."""
    filepath = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_FILE
    importer = DataImporter()

    try:
        importer.import_file(filepath)
    except Exception as e:
        print("\n--- ERROR ---")
        print(e)
        print("-------------")
        input("Press ENTER to exit...")


if __name__ == "__main__":
    main()
