"""Bloomberg DAPI wide-format close-price file reader."""

from __future__ import annotations

from dataclasses import dataclass, field
import io

import pandas as pd

from modules.shared.tickers import canonical_ticker

_PRICE_COLUMNS = ["ticker", "price_date", "close_price", "source"]


@dataclass(frozen=True)
class PriceFileParseResult:
    """Parsed price rows plus what the parser had to discard."""

    frame: pd.DataFrame
    skipped_tickers: list[str] = field(default_factory=list)
    skipped_rows: int = 0

    @property
    def is_empty(self) -> bool:
        return self.frame.empty


class BloombergPriceFileReader:
    """
    Parse a Bloomberg DAPI wide-format close price Excel file.

    Expected layout:
        Row 0 : metadata / title row — ignored
        Row 1 : column A blank or label; columns B+ = ticker symbols
        Row 2+ : column A = date (Excel date or YYYY-MM-DD string); columns B+ = close prices

    Ticker headers are canonicalized (``AAPL US Equity`` -> ``AAPL``) so that stored
    prices match the identifiers the rest of the application uses. Headers that
    cannot be canonicalized are reported in ``skipped_tickers`` rather than dropped
    silently.
    """

    _TICKER_ROW: int = 1
    _DATA_START_ROW: int = 2

    def read(self, file_content: bytes) -> PriceFileParseResult:
        """
        Parse the file into long rows ['ticker', 'price_date', 'close_price', 'source'].

        'source' is always 'bloomberg'. Rows with an unparseable date or a
        non-numeric price are dropped and counted in ``skipped_rows``.
        """
        raw = pd.read_excel(io.BytesIO(file_content), header=None)
        if raw.empty or raw.shape[0] <= self._DATA_START_ROW or raw.shape[1] < 2:
            return PriceFileParseResult(frame=self._empty_frame())

        header_values = raw.iloc[self._TICKER_ROW, 1:].tolist()
        canonical_by_column: dict[int, str] = {}
        skipped_tickers: list[str] = []
        for offset, header in enumerate(header_values):
            canonical = canonical_ticker(header)
            if canonical:
                canonical_by_column[offset + 1] = canonical
            elif str(header or "").strip():
                skipped_tickers.append(str(header).strip())

        if not canonical_by_column:
            return PriceFileParseResult(
                frame=self._empty_frame(),
                skipped_tickers=skipped_tickers,
            )

        column_indexes = sorted(canonical_by_column)
        data = raw.iloc[self._DATA_START_ROW:, [0] + column_indexes].copy()
        data.columns = ["date"] + [canonical_by_column[idx] for idx in column_indexes]

        parsed_dates = pd.to_datetime(data["date"], errors="coerce")
        rows_with_bad_dates = int(parsed_dates.isna().sum()) * len(column_indexes)
        data = data.loc[parsed_dates.notna()].copy()
        if data.empty:
            return PriceFileParseResult(
                frame=self._empty_frame(),
                skipped_tickers=skipped_tickers,
                skipped_rows=rows_with_bad_dates,
            )
        data["price_date"] = parsed_dates.loc[data.index].dt.strftime("%Y-%m-%d")

        long = data.drop(columns=["date"]).melt(
            id_vars="price_date",
            var_name="ticker",
            value_name="close_price",
        )
        long["close_price"] = pd.to_numeric(long["close_price"], errors="coerce")
        dropped_prices = int(long["close_price"].isna().sum())
        long = long.dropna(subset=["close_price"])
        long["source"] = "bloomberg"

        # Duplicate headers for the same security collapse onto one canonical
        # ticker; keep the last occurrence so the right-most column wins.
        long = long.drop_duplicates(subset=["ticker", "price_date"], keep="last")

        return PriceFileParseResult(
            frame=long[_PRICE_COLUMNS].reset_index(drop=True),
            skipped_tickers=skipped_tickers,
            skipped_rows=rows_with_bad_dates + dropped_prices,
        )

    @staticmethod
    def _empty_frame() -> pd.DataFrame:
        return pd.DataFrame(columns=_PRICE_COLUMNS)
