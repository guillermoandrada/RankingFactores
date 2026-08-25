"""Price data management service."""

from __future__ import annotations

from modules.infrastructure.db import FinancialDatabase
from modules.infrastructure.ingestion.readers.bloomberg_prices import BloombergPriceFileReader
from modules.infrastructure.market_data import (
    BasePriceProvider,
    fetch_latest_adjusted_closes,
)
from modules.shared.tickers import canonical_ticker_map


class PriceService:
    """Ingest, inspect and delete cached security prices."""

    def __init__(
        self,
        db: FinancialDatabase,
        price_provider: BasePriceProvider | None = None,
    ) -> None:
        self._db = db
        self._price_provider = price_provider
        self._reader = BloombergPriceFileReader()

    def ingest_from_file(self, file_content: bytes, filename: str) -> dict:
        """Parse an uploaded Bloomberg price file and upsert rows into the DB."""
        parsed = self._reader.read(file_content)
        if parsed.is_empty:
            raise ValueError(
                f"No valid price rows found in '{filename}'. "
                "Expected format: row 1 = ticker headers (column B onwards), "
                "row 2+ = date (column A) + close prices."
            )
        frame = parsed.frame
        rows_written = self._db.upsert_price_data(frame)
        return {
            "tickers_imported": sorted(frame["ticker"].unique().tolist()),
            "tickers_skipped": sorted(parsed.skipped_tickers),
            "rows_written": rows_written,
            "rows_skipped": parsed.skipped_rows,
            "date_range": {
                "min": frame["price_date"].min(),
                "max": frame["price_date"].max(),
            },
        }

    def list_cached_tickers(self) -> list[dict]:
        """Return all tickers with cached prices and their date ranges."""
        return self._db.list_cached_tickers()

    def delete_tickers(self, tickers: list[str]) -> dict:
        """Delete all cached prices for the given tickers."""
        canonical = sorted(set(canonical_ticker_map(list(tickers)).values()))
        if not canonical:
            raise ValueError("No valid tickers provided.")
        deleted = self._db.delete_price_data_for_tickers(canonical)
        return {"deleted_rows": deleted, "tickers": canonical}

    def get_latest_closes(self, tickers: list[str]) -> dict:
        """
        Return the most recent adjusted close per ticker.

        Resolution goes through the configured price provider, so manually
        uploaded prices take priority over Yahoo Finance exactly as they do in
        backtests and IC analysis.
        """
        result = fetch_latest_adjusted_closes(tickers, provider=self._price_provider)
        return {
            "closes": result.closes,
            "resolved_identifiers": result.resolved_identifiers,
            "missing_identifiers": result.missing_identifiers,
        }
