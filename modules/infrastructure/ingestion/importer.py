"""
Orchestrates reading, validation, and persistence of financial data.
"""

import json
import os
import sys
from typing import Optional

import pandas as pd

from modules.config import FIXED_COLUMNS, DEFAULT_INPUT_FILE
from modules.infrastructure.db import FinancialDatabase
from modules.infrastructure.ingestion.file_reader import FileReader
from modules.domain.models import ImportResult


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

    def import_file(
        self,
        filepath: str,
        verbose: bool = True,
        period_override: Optional[str] = None,
        reader: str = "bloomberg",
        if_period_exists: str = "replace",
        index_code_override: Optional[str] = None,
    ) -> ImportResult:
        """
        Import a single file into the database.
        Returns ImportResult with counts.
        period_override: use instead of extracting from file.
        reader: concrete reader name such as 'bloomberg' or 'reuters_metrics'.
        if_period_exists: 'replace' (overwrite) or 'append' (merge new metrics/securities).
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        df = self._reader.read(filepath, reader)
        if verbose:
            print(f"Read file: {filepath}")

        df = df.dropna(subset=["Ticker"])
        period = period_override or self._reader.extract_period(filepath, reader)
        if not str(period or "").strip() or str(period).strip().upper() == "UNKNOWN":
            raise ValueError(
                f"Could not determine period for reader '{reader}'. "
                "Provide a period manually."
            )
        if reader == "bql":
            df = self._enrich_bql_dataframe(df, period)

        self._validate_columns(df)

        index_code = index_code_override or self._reader.extract_index_code(filepath, reader)
        if verbose:
            print(f"--> Index code detected: {index_code}")
        if verbose:
            print(f"--> Period: {period}")

        mode = "append" if if_period_exists == "append" else "replace"
        result = self._db.save_fundamentals(
            df,
            period,
            index_code,
            mode=mode,
            preserve_existing_classification=(
                reader == "reuters_metrics" and mode == "append"
            ),
        )

        if verbose:
            print(
                f"Success! Imported {result.companies_count} companies, "
                f"{result.metrics_count} metrics, {result.records_count} records."
            )
        return result

    def describe_file(
        self,
        filepath: str,
        *,
        reader: str = "bloomberg",
        period_override: Optional[str] = None,
        sample_row_count: int = 5,
    ) -> dict:
        """
        Describe what importing this file would produce, without touching the database.

        Uses the same readers as import_file, so the reported period and index code are
        the ones an import would actually use. Detection problems are returned as
        `period_error` rather than raised: for a preview they are the answer, not a failure.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        df = self._reader.read(filepath, reader)
        if "Ticker" in df.columns:
            df = df.dropna(subset=["Ticker"])

        detected_period: Optional[str] = None
        period_error: Optional[str] = None
        try:
            candidate = self._reader.extract_period(filepath, reader)
            if str(candidate or "").strip() and str(candidate).strip().upper() != "UNKNOWN":
                detected_period = str(candidate).strip()
            else:
                period_error = f"Reader '{reader}' could not determine a period from this file."
        except (ValueError, KeyError, IndexError, OSError) as exc:
            period_error = str(exc)

        index_code: Optional[str] = None
        try:
            index_code = self._reader.extract_index_code(filepath, reader)
        except (ValueError, KeyError, IndexError, OSError):
            index_code = None

        period = (period_override or "").strip() or detected_period

        return {
            "reader": reader,
            "period": period,
            "period_source": "manual" if period_override else "file",
            "detected_period": detected_period,
            "period_error": period_error if not period_override else None,
            "index_code": index_code,
            "sheet_names": self._sheet_names(filepath),
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "columns": [str(column) for column in df.columns],
            "sample_rows": self._sample_rows(df, sample_row_count),
            "missing_ticker_column": "Ticker" not in df.columns,
        }

    @staticmethod
    def _sheet_names(filepath: str) -> list[str]:
        try:
            with pd.ExcelFile(filepath) as workbook:
                return [str(name) for name in workbook.sheet_names]
        except (ValueError, OSError):
            return []

    @staticmethod
    def _sample_rows(df: pd.DataFrame, count: int) -> list[dict]:
        """JSON-safe records: to_json handles NaN and timestamps that to_dict does not."""
        if df.empty or count <= 0:
            return []
        return json.loads(df.head(count).to_json(orient="records", date_format="iso"))

    def _enrich_bql_dataframe(self, df: pd.DataFrame, period: str) -> pd.DataFrame:
        tickers = df["Ticker"].dropna().astype(str).str.strip()
        unique_tickers = [ticker for ticker in dict.fromkeys(tickers.tolist()) if ticker]
        metadata_rows = self._db.get_security_metadata(period=period, tickers=unique_tickers)
        metadata_by_ticker = {str(row["ticker"]).strip(): row for row in metadata_rows}

        enriched = df.copy()
        enriched["Market Cap (USD)"] = enriched["Ticker"].map(
            lambda ticker: metadata_by_ticker.get(str(ticker).strip(), {}).get("market_cap_usd")
        )

        missing_metadata: list[str] = []
        for ticker in unique_tickers:
            matching_rows = enriched[enriched["Ticker"].astype(str).str.strip() == ticker]
            row_values = matching_rows.iloc[0] if not matching_rows.empty else None
            row = metadata_by_ticker.get(ticker, {})
            if (
                row_values is None
                or not self._has_non_empty_value(row_values.get("Long Name"))
                or not self._has_non_empty_value(row_values.get("GICS Sector Name"))
                or not self._has_non_empty_value(row_values.get("GICS Industry Group Name"))
            ):
                missing_metadata.append(ticker)

        if missing_metadata:
            raise ValueError(
                "BQL upload requires Name and Classification metadata for "
                f"every ticker. Incomplete metadata for: {sorted(missing_metadata)}"
            )

        return enriched

    def _has_non_empty_value(self, value: object) -> bool:
        if value is None or pd.isna(value):
            return False
        return bool(str(value).strip())

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
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as e:
        print("\n--- ERROR ---")
        print(e)
        print("-------------")
        input("Press ENTER to exit...")


if __name__ == "__main__":
    main()
