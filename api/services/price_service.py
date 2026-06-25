"""Price data management service."""

from __future__ import annotations

from modules.infrastructure.db import FinancialDatabase
from modules.infrastructure.ingestion.readers.bloomberg_prices import BloombergPriceFileReader


class PriceService:
    def __init__(self, db: FinancialDatabase) -> None:
        self._db = db
        self._reader = BloombergPriceFileReader()

    def ingest_from_file(self, file_content: bytes, filename: str) -> dict:
        """Parse an uploaded Bloomberg price file and upsert rows into the DB."""
        df = self._reader.read(file_content)
        if df.empty:
            raise ValueError(
                f"No valid price rows found in '{filename}'. "
                "Expected format: row 1 = ticker headers (column B onwards), "
                "row 2+ = date (column A) + close prices."
            )
        rows_written = self._db.upsert_price_data(df)
        tickers = sorted(df["ticker"].unique().tolist())
        return {
            "tickers_imported": tickers,
            "rows_written": rows_written,
            "date_range": {
                "min": df["price_date"].min(),
                "max": df["price_date"].max(),
            },
        }

    def list_cached_tickers(self) -> list[dict]:
        """Return all tickers with cached prices and their date ranges."""
        return self._db.list_cached_tickers()

    def delete_tickers(self, tickers: list[str]) -> dict:
        """Delete all cached prices for the given tickers."""
        deleted = self._db.delete_price_data_for_tickers(tickers)
        return {"deleted_rows": deleted, "tickers": tickers}
