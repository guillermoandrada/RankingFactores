"""Database metric (variable) ingestion service."""

from __future__ import annotations

import pandas as pd

from modules.infrastructure.db import FinancialDatabase
from modules.infrastructure.ingestion.readers.bloomberg_individual_variable import (
    BloombergIndividualVariableReader,
)

_EXPECTED_LAYOUT = (
    "Expected layout: row 1 = index name and period date (A1, B1), row 2 = field labels, "
    "row 3+ = either ticker, long name, GICS sector, GICS industry group, value (five "
    "columns per period) or ticker, value (two columns per period). The sheet name is "
    "the variable name."
)


class DbMetricService:
    """Ingest database-backed metrics from a Bloomberg individual-variable file."""

    def __init__(self, db: FinancialDatabase) -> None:
        self._db = db
        self._reader = BloombergIndividualVariableReader()

    def ingest_variable_file(
        self,
        file_content: bytes,
        filename: str,
        *,
        sheet_name: str | None = None,
    ) -> dict:
        """
        Parse a single-variable Bloomberg file and import every period it holds.

        The variable name is the sheet name. Each period is imported in append mode, so
        the variable replaces its own previous values while the period's other metrics
        stay put. Periods are written one at a time; the reported periods are those
        committed.

        What the file carries decides how far the import may go:

        * ticker, long name, GICS sector, GICS industry group, value — enough to create a
          security, so unknown tickers are added and classifications are refreshed.
        * ticker, value — not enough, so the import is restricted to securities that
          already exist and unknown tickers come back in ``securities_skipped``. No
          security is created and no classification is touched.

        The index name of each period is reported back for checking only: a variable
        file may cover part of a universe, so it never rewrites index membership.
        """
        parsed = self._reader.read(file_content, sheet_name=sheet_name)
        if parsed.is_empty:
            raise ValueError(
                f"No valid variable values found in '{filename}'. {_EXPECTED_LAYOUT}"
            )

        frames = parsed.frames
        securities_skipped: list[str] = []
        if not parsed.creates_securities:
            frames, securities_skipped = self._restrict_to_existing_securities(frames)
            if not frames:
                raise ValueError(
                    f"'{filename}' holds ticker and value only, and none of its "
                    f"{len(securities_skipped)} identifiers match a security in the "
                    "database. Add long name, GICS sector and GICS industry group "
                    "columns to create the securities, or import the period "
                    "fundamentals first."
                )

        periods: list[dict] = []
        for period in sorted(frames):
            result = self._db.save_fundamentals(
                frames[period],
                period,
                mode="append",
                preserve_existing_classification=not parsed.creates_securities,
            )
            periods.append(
                {
                    "period": result.period,
                    "index_code": parsed.index_codes.get(period),
                    "companies_count": result.companies_count,
                    "metrics_count": result.metrics_count,
                    "records_count": result.records_count,
                }
            )

        securities = {
            ticker for frame in frames.values() for ticker in frame["Ticker"].tolist()
        }
        return {
            "variable": parsed.variable_name,
            "creates_securities": parsed.creates_securities,
            "periods": periods,
            "securities_count": len(securities),
            "records_count": sum(period["records_count"] for period in periods),
            "securities_skipped": securities_skipped,
            "periods_skipped": parsed.skipped_periods,
            "rows_skipped": parsed.skipped_rows,
        }

    def _restrict_to_existing_securities(
        self,
        frames: dict[str, pd.DataFrame],
    ) -> tuple[dict[str, pd.DataFrame], list[str]]:
        """
        Drop rows whose security is not in the database yet.

        This filter is what keeps a ticker-and-value file from creating anything:
        ``save_fundamentals`` upserts every ticker it is given, so the guarantee has to
        be made here, before the write.
        """
        requested = {
            ticker for frame in frames.values() for ticker in frame["Ticker"].tolist()
        }
        existing = self._db.get_existing_tickers(sorted(requested))

        kept: dict[str, pd.DataFrame] = {}
        for period, frame in frames.items():
            known_rows = frame[frame["Ticker"].isin(existing)]
            if not known_rows.empty:
                kept[period] = known_rows.reset_index(drop=True)
        return kept, sorted(requested - existing)
