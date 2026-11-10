"""Bloomberg BQL block-format total-return file reader."""

from __future__ import annotations

from dataclasses import dataclass, field
import io

import pandas as pd

from modules.shared.tickers import canonical_ticker

_TOTAL_RETURN_COLUMNS = [
    "ticker",
    "start_date",
    "end_date",
    "horizon_months",
    "total_return",
    "source",
]

_AVERAGE_DAYS_PER_MONTH = 30.4375


@dataclass(frozen=True)
class TotalReturnFileParseResult:
    """Parsed total-return rows plus what the parser had to discard."""

    frame: pd.DataFrame
    skipped_tickers: list[str] = field(default_factory=list)
    skipped_rows: int = 0

    @property
    def is_empty(self) -> bool:
        return self.frame.empty


class BloombergTotalReturnFileReader:
    """
    Parse a Bloomberg BQL total-return export laid out in side-by-side blocks.

    Every sheet may hold any number of five-column blocks, one per return window:

        Row 0 : col +0 = benchmark label (ignored); col +1 = window end date;
                col +2 = window start date
        Row 1 : col +0 = "ID"; col +4 = ``total_return(...)`` field header
        Row 2+: col +0 = Bloomberg identifier; cols +1..+3 = name / sector /
                industry (ignored); col +4 = total return as a decimal fraction

    Sheets without such blocks (formula helpers, notes) are ignored. The horizon
    is derived from the window dates rather than the sheet name, so a renamed
    sheet cannot mislabel its returns.
    """

    _DATE_ROW: int = 0
    _HEADER_ROW: int = 1
    _DATA_START_ROW: int = 2
    _BLOCK_WIDTH: int = 5
    _END_DATE_OFFSET: int = 1
    _START_DATE_OFFSET: int = 2
    _VALUE_OFFSET: int = 4
    _ID_HEADER: str = "ID"
    _VALUE_HEADER_PREFIX: str = "total_return"

    def read(self, file_content: bytes) -> TotalReturnFileParseResult:
        """
        Parse every block into long rows
        ['ticker', 'start_date', 'end_date', 'horizon_months', 'total_return', 'source'].

        'source' is always 'bloomberg'. Rows with a non-numeric return are dropped
        and counted in ``skipped_rows``.
        """
        sheets = pd.read_excel(io.BytesIO(file_content), header=None, sheet_name=None)
        frames: list[pd.DataFrame] = []
        skipped_tickers: set[str] = set()
        skipped_rows = 0
        for raw in sheets.values():
            for start_column in self._block_start_columns(raw):
                block, block_skipped_tickers, block_skipped_rows = self._read_block(
                    raw, start_column
                )
                frames.append(block)
                skipped_tickers.update(block_skipped_tickers)
                skipped_rows += block_skipped_rows

        non_empty = [frame for frame in frames if not frame.empty]
        if not non_empty:
            return TotalReturnFileParseResult(
                frame=self._empty_frame(),
                skipped_tickers=sorted(skipped_tickers),
                skipped_rows=skipped_rows,
            )

        # A window exported twice collapses onto the right-most occurrence.
        combined = pd.concat(non_empty, ignore_index=True).drop_duplicates(
            subset=["ticker", "start_date", "end_date"], keep="last"
        )
        return TotalReturnFileParseResult(
            frame=combined[_TOTAL_RETURN_COLUMNS].reset_index(drop=True),
            skipped_tickers=sorted(skipped_tickers),
            skipped_rows=skipped_rows,
        )

    def _block_start_columns(self, raw: pd.DataFrame) -> list[int]:
        """Return the first column of every block whose headers match the layout."""
        if raw.shape[0] <= self._DATA_START_ROW:
            return []
        last_start = raw.shape[1] - self._BLOCK_WIDTH
        return [
            column
            for column in range(0, last_start + 1)
            if self._is_block_start(raw, column)
        ]

    def _is_block_start(self, raw: pd.DataFrame, column: int) -> bool:
        id_header = str(raw.iat[self._HEADER_ROW, column]).strip()
        value_header = str(raw.iat[self._HEADER_ROW, column + self._VALUE_OFFSET]).strip()
        return id_header == self._ID_HEADER and value_header.startswith(
            self._VALUE_HEADER_PREFIX
        )

    def _read_block(
        self, raw: pd.DataFrame, start_column: int
    ) -> tuple[pd.DataFrame, list[str], int]:
        """Parse one block into long rows, its unusable identifiers and dropped rows."""
        end_date = pd.to_datetime(
            raw.iat[self._DATE_ROW, start_column + self._END_DATE_OFFSET], errors="coerce"
        )
        start_date = pd.to_datetime(
            raw.iat[self._DATE_ROW, start_column + self._START_DATE_OFFSET], errors="coerce"
        )
        data = raw.iloc[
            self._DATA_START_ROW :, [start_column, start_column + self._VALUE_OFFSET]
        ]
        data.columns = ["identifier", "total_return"]
        data = data.dropna(subset=["identifier"])
        data = data.loc[data["identifier"].astype(str).str.strip() != ""]
        if pd.isna(end_date) or pd.isna(start_date) or end_date <= start_date:
            return self._empty_frame(), [], len(data)

        data = data.assign(ticker=data["identifier"].map(canonical_ticker))
        unusable = data.loc[data["ticker"] == "", "identifier"]
        skipped_tickers = [str(value).strip() for value in unusable]
        data = data.loc[data["ticker"] != ""].copy()

        data["total_return"] = pd.to_numeric(data["total_return"], errors="coerce")
        dropped_returns = int(data["total_return"].isna().sum())
        data = data.dropna(subset=["total_return"])

        data["start_date"] = start_date.strftime("%Y-%m-%d")
        data["end_date"] = end_date.strftime("%Y-%m-%d")
        data["horizon_months"] = self._horizon_months(start_date, end_date)
        data["source"] = "bloomberg"
        return data[_TOTAL_RETURN_COLUMNS], skipped_tickers, len(skipped_tickers) + dropped_returns

    @staticmethod
    def _horizon_months(start_date: pd.Timestamp, end_date: pd.Timestamp) -> int:
        """Round the window length to whole months (364 or 366 days -> 12)."""
        return round((end_date - start_date).days / _AVERAGE_DAYS_PER_MONTH)

    @staticmethod
    def _empty_frame() -> pd.DataFrame:
        return pd.DataFrame(columns=_TOTAL_RETURN_COLUMNS)
