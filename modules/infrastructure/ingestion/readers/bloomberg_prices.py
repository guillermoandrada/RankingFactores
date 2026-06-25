"""Bloomberg DAPI wide-format close-price file reader."""

from __future__ import annotations

import io

import pandas as pd


class BloombergPriceFileReader:
    """
    Parse a Bloomberg DAPI wide-format close price Excel file.

    Expected layout:
        Row 0 : metadata / title row — ignored
        Row 1 : column A blank or label; columns B+ = ticker symbols
        Row 2+ : column A = date (Excel date or YYYY-MM-DD string); columns B+ = close prices
    """

    _TICKER_ROW: int = 1
    _DATA_START_ROW: int = 2

    def read(self, file_content: bytes) -> pd.DataFrame:
        """
        Parse the file and return a long DataFrame with columns:
        ['ticker', 'price_date', 'close_price', 'source'].
        'source' is always 'bloomberg'. Rows with missing dates or prices are dropped.
        """
        raw = pd.read_excel(io.BytesIO(file_content), header=None)

        ticker_values = raw.iloc[self._TICKER_ROW, 1:].tolist()
        tickers = [
            str(t).strip()
            for t in ticker_values
            if str(t or "").strip() not in ("", "nan", "None")
        ]
        if not tickers:
            return pd.DataFrame(columns=["ticker", "price_date", "close_price", "source"])

        n_ticker_cols = len(tickers)
        data = raw.iloc[self._DATA_START_ROW:, : n_ticker_cols + 1].copy()

        col_names = ["date"] + [
            str(raw.iloc[self._TICKER_ROW, col_idx])
            for col_idx in range(1, n_ticker_cols + 1)
        ]
        data.columns = col_names

        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data = data.dropna(subset=["date"])
        if data.empty:
            return pd.DataFrame(columns=["ticker", "price_date", "close_price", "source"])

        data["price_date"] = data["date"].dt.strftime("%Y-%m-%d")

        price_cols = [c for c in data.columns if c not in ("date", "price_date")]
        long = data[["price_date"] + price_cols].melt(
            id_vars="price_date",
            value_vars=price_cols,
            var_name="ticker",
            value_name="close_price",
        )
        long["ticker"] = long["ticker"].str.strip()
        long["close_price"] = pd.to_numeric(long["close_price"], errors="coerce")
        long = long.dropna(subset=["close_price"])
        long = long[long["ticker"].str.len() > 0]
        long["source"] = "bloomberg"

        return long[["ticker", "price_date", "close_price", "source"]].reset_index(drop=True)
