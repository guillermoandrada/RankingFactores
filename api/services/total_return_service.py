"""Total-return data management service."""

from __future__ import annotations

from modules.infrastructure.db import FinancialDatabase
from modules.infrastructure.ingestion.readers.bloomberg_total_returns import (
    BloombergTotalReturnFileReader,
)


class TotalReturnService:
    """Ingest, inspect and delete stored security total returns."""

    def __init__(self, db: FinancialDatabase) -> None:
        self._db = db
        self._reader = BloombergTotalReturnFileReader()

    def ingest_from_file(self, file_content: bytes, filename: str) -> dict:
        """Parse an uploaded Bloomberg total-return file and upsert rows into the DB."""
        parsed = self._reader.read(file_content)
        if parsed.is_empty:
            raise ValueError(
                f"No valid total-return rows found in '{filename}'. "
                "Expected blocks of five columns: row 1 = benchmark, end date, start date; "
                "row 2 = 'ID', name, sector, industry, 'total_return(...)'; "
                "row 3+ = Bloomberg identifier + total return."
            )
        frame = parsed.frame
        rows_written = self._db.upsert_total_returns(frame)
        return {
            "tickers_imported": sorted(frame["ticker"].unique().tolist()),
            "tickers_skipped": sorted(parsed.skipped_tickers),
            "rows_written": rows_written,
            "rows_skipped": parsed.skipped_rows,
            "horizons_months": sorted(int(h) for h in frame["horizon_months"].unique()),
            "windows_imported": int(
                frame[["start_date", "end_date"]].drop_duplicates().shape[0]
            ),
            "date_range": {
                "min": frame["start_date"].min(),
                "max": frame["end_date"].max(),
            },
        }

    def list_windows(self) -> list[dict]:
        """Return every stored return window with its ticker coverage."""
        return self._db.list_total_return_windows()

    def delete_windows(self, horizon_months: int | None = None) -> dict:
        """Delete stored total returns, optionally only those of one horizon."""
        if horizon_months is not None and horizon_months <= 0:
            raise ValueError("horizon_months must be a positive number of months.")
        deleted = self._db.delete_total_returns(horizon_months=horizon_months)
        return {"deleted_rows": deleted, "horizon_months": horizon_months}
