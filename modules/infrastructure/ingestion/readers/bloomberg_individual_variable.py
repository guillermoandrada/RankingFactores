"""Bloomberg individual-variable file reader: one variable across many periods."""

from __future__ import annotations

from dataclasses import dataclass, field
import io

import pandas as pd

from modules.config import FIXED_COLUMNS
from modules.infrastructure.ingestion.readers.bloomberg import _format_date_as_period
from modules.shared.tickers import ticker_from_bloomberg_id

_DESCRIPTIVE_COLUMNS = ("Long Name", "GICS Sector Name", "GICS Industry Group Name")

# The two supported block layouts, as column name -> offset from the block's first column.
# "value" is a placeholder for the column named after the sheet.
_VALUES_ONLY_WIDTH = 2
_ENRICHED_WIDTH = 5
_BLOCK_LAYOUTS: dict[int, dict[str, int]] = {
    _VALUES_ONLY_WIDTH: {"Ticker": 0, "value": 1},
    _ENRICHED_WIDTH: {
        "Ticker": 0,
        "Long Name": 1,
        "GICS Sector Name": 2,
        "GICS Industry Group Name": 3,
        "value": 4,
    },
}
_VALUE_PLACEHOLDER = "value"


def _cell_text(value: object) -> str:
    """Return a trimmed cell as text, or '' for an empty cell."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


@dataclass(frozen=True)
class IndividualVariableParseResult:
    """One importable table per period, plus what the parser had to discard."""

    variable_name: str
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    index_codes: dict[str, str] = field(default_factory=dict)
    creates_securities: bool = False
    skipped_periods: list[str] = field(default_factory=list)
    skipped_rows: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.frames


@dataclass(frozen=True)
class _PeriodBlock:
    """One period's columns inside the sheet."""

    period: str
    index_code: str
    start_column: int
    width: int


class BloombergIndividualVariableReader:
    """
    Parse a Bloomberg workbook holding a single variable observed at several periods.

    The sheet name is the variable name. Row 0 carries the index name over the ticker
    column and the period date in the next column; row 1 holds vendor field labels and is
    ignored, so Excel errors there are harmless; data starts on row 2. Two block layouts
    are supported, and the file itself decides which:

        5 columns  ticker, long name, GICS sector, GICS industry group, value
        2 columns  ticker, value

    The wide form carries everything needed to create a security, so ``creates_securities``
    comes back True. The narrow form does not, and the caller must restrict the import to
    securities that already exist.

    Every block yields a table shaped like any other fundamentals import (``FIXED_COLUMNS``
    plus one metric column named after the sheet); in the narrow form the descriptive
    columns are blank. Blocks whose date cannot be read are reported in ``skipped_periods``
    rather than dropped silently.
    """

    _HEADER_ROW: int = 0
    _DATA_START_ROW: int = 2

    def read(
        self,
        file_content: bytes,
        sheet_name: str | None = None,
    ) -> IndividualVariableParseResult:
        """
        Parse the sheet into one import-ready frame per period.

        ``sheet_name`` defaults to the first sheet in the workbook. Rows without a
        ticker are dropped and counted in ``skipped_rows``; a row with a ticker but no
        numeric value is kept, so the security still joins the period.
        """
        workbook = pd.ExcelFile(io.BytesIO(file_content))
        resolved_sheet = self._resolve_sheet_name(workbook, sheet_name)
        variable_name = str(resolved_sheet).strip()
        if variable_name in FIXED_COLUMNS:
            raise ValueError(
                f"Sheet name '{variable_name}' is a reserved column name. "
                "Rename the sheet to the variable it holds."
            )

        raw = workbook.parse(sheet_name=resolved_sheet, header=None)
        if raw.shape[0] <= self._DATA_START_ROW or raw.shape[1] < _VALUES_ONLY_WIDTH:
            return IndividualVariableParseResult(variable_name=variable_name)

        blocks, skipped_periods = self._detect_period_blocks(raw)

        frames: dict[str, pd.DataFrame] = {}
        index_codes: dict[str, str] = {}
        skipped_rows = 0
        for block in blocks:
            values, block_skipped_rows = self._read_block(raw, block, variable_name)
            skipped_rows += block_skipped_rows
            if values.empty:
                continue
            # A period repeated across blocks collapses onto the right-most one.
            frames[block.period] = values
            if block.index_code:
                index_codes[block.period] = block.index_code

        return IndividualVariableParseResult(
            variable_name=variable_name,
            frames=frames,
            index_codes=index_codes,
            creates_securities=bool(blocks) and blocks[0].width == _ENRICHED_WIDTH,
            skipped_periods=skipped_periods,
            skipped_rows=skipped_rows,
        )

    def _resolve_sheet_name(self, workbook: pd.ExcelFile, sheet_name: str | None) -> str:
        requested = str(sheet_name or "").strip()
        if not requested:
            return str(workbook.sheet_names[0])

        available = {str(name).strip().lower(): str(name) for name in workbook.sheet_names}
        match = available.get(requested.lower())
        if match is None:
            raise ValueError(
                f"Sheet '{sheet_name}' not found. Available sheets: {list(workbook.sheet_names)}"
            )
        return match

    def _detect_period_blocks(self, raw: pd.DataFrame) -> tuple[list[_PeriodBlock], list[str]]:
        """
        Locate every period block, its date and the layout in use.

        A block starts wherever the next header cell parses as a date, which is true of
        both layouts, so the distance between consecutive blocks gives the width. A lone
        block has no such distance and is measured by :meth:`_infer_single_block_width`.
        """
        start_columns = [
            column
            for column in range(raw.shape[1] - 1)
            if _format_date_as_period(raw.iat[self._HEADER_ROW, column + 1])
        ]
        if not start_columns:
            return [], self._report_unreadable_dates(raw, first_column=0, width=_ENRICHED_WIDTH)

        width = self._resolve_block_width(raw, start_columns)

        blocks: list[_PeriodBlock] = []
        skipped_periods: list[str] = []
        for column in start_columns:
            period = _format_date_as_period(raw.iat[self._HEADER_ROW, column + 1])
            remaining = raw.shape[1] - column
            if remaining < width:
                skipped_periods.append(f"{period} (only {remaining} column(s) left)")
                continue
            self._verify_value_column(raw, start_column=column, width=width, period=period)
            blocks.append(
                _PeriodBlock(
                    period=period,
                    index_code=_cell_text(raw.iat[self._HEADER_ROW, column]),
                    start_column=column,
                    width=width,
                )
            )

        skipped_periods.extend(
            self._report_unreadable_dates(
                raw,
                first_column=start_columns[0],
                width=width,
                known_columns=set(start_columns),
            )
        )
        return blocks, skipped_periods

    def _verify_value_column(
        self,
        raw: pd.DataFrame,
        *,
        start_column: int,
        width: int,
        period: str,
    ) -> None:
        """
        Fail loudly when a block's numbers are not in the column the layout expects.

        The gap between two blocks measures the one on its *left*, so the last block in a
        sheet is the one place a different layout could slip through. Without this check it
        would be read from whichever columns the assumed width points at.
        """
        value_column = start_column + _BLOCK_LAYOUTS[width][_VALUE_PLACEHOLDER]
        if self._holds_numbers(raw, value_column):
            return

        other_width = _ENRICHED_WIDTH if width == _VALUES_ONLY_WIDTH else _VALUES_ONLY_WIDTH
        other_column = start_column + _BLOCK_LAYOUTS[other_width][_VALUE_PLACEHOLDER]
        if other_column < raw.shape[1] and self._holds_numbers(raw, other_column):
            raise ValueError(
                f"Period {period} keeps its values in sheet column {other_column + 1} "
                f"rather than column {value_column + 1}, where the {width}-column layout "
                "expects them. Use one layout for the whole sheet."
            )

    def _holds_numbers(self, raw: pd.DataFrame, column: int) -> bool:
        """True when a data column has at least one numeric cell."""
        data = raw.iloc[self._DATA_START_ROW:, column]
        return bool(pd.to_numeric(data, errors="coerce").notna().any())

    def _resolve_block_width(self, raw: pd.DataFrame, start_columns: list[int]) -> int:
        """Width of every period block, rejecting layouts this reader cannot read."""
        gaps = {
            second - first for first, second in zip(start_columns, start_columns[1:])
        }
        if len(gaps) > 1:
            raise ValueError(
                "Period blocks have inconsistent widths "
                f"({sorted(gaps)} columns apart). Use one layout for the whole sheet: "
                "ticker and value, or ticker, long name, GICS sector, GICS industry "
                "group and value."
            )

        width = gaps.pop() if gaps else self._infer_single_block_width(raw, start_columns[0])
        if width not in _BLOCK_LAYOUTS:
            raise ValueError(
                f"Each period block spans {width} columns, which is neither "
                f"{_VALUES_ONLY_WIDTH} (ticker, value) nor {_ENRICHED_WIDTH} "
                "(ticker, long name, GICS sector, GICS industry group, value). "
                "Remove blank or extra columns between periods."
            )
        return width

    def _infer_single_block_width(self, raw: pd.DataFrame, start_column: int) -> int:
        """
        Measure a lone block from its second column.

        In the narrow layout that column holds the values, in the wide one the security
        name — and a name never reads as a number.
        """
        if start_column + _ENRICHED_WIDTH > raw.shape[1]:
            return _VALUES_ONLY_WIDTH
        if self._holds_numbers(raw, start_column + 1):
            return _VALUES_ONLY_WIDTH
        return _ENRICHED_WIDTH

    def _report_unreadable_dates(
        self,
        raw: pd.DataFrame,
        *,
        first_column: int,
        width: int,
        known_columns: set[int] | None = None,
    ) -> list[str]:
        """Name the block slots that carry a header but no readable date."""
        known = known_columns or set()
        unreadable: list[str] = []
        for column in range(first_column, raw.shape[1] - 1, width):
            if column in known:
                continue
            date_text = _cell_text(raw.iat[self._HEADER_ROW, column + 1])
            index_text = _cell_text(raw.iat[self._HEADER_ROW, column])
            if date_text or index_text:
                unreadable.append(date_text or f"column {column + 2}")
        return unreadable

    def _read_block(
        self,
        raw: pd.DataFrame,
        block: _PeriodBlock,
        variable_name: str,
    ) -> tuple[pd.DataFrame, int]:
        """Return one period's import-ready table and how many rows were dropped."""
        layout = _BLOCK_LAYOUTS[block.width]
        columns = [block.start_column + offset for offset in layout.values()]
        body = raw.iloc[self._DATA_START_ROW:, columns].copy()
        populated = body.notna().any(axis=1)
        body.columns = [
            variable_name if name == _VALUE_PLACEHOLDER else name for name in layout
        ]

        body["Ticker"] = body["Ticker"].map(ticker_from_bloomberg_id)
        for column in _DESCRIPTIVE_COLUMNS:
            # Absent in the narrow layout: leave them blank so the importer, which only
            # fills empty fields, cannot overwrite what the security already has.
            body[column] = body[column].map(_cell_text) if column in body else ""
        body[variable_name] = pd.to_numeric(body[variable_name], errors="coerce")

        has_ticker = body["Ticker"].ne("")
        skipped_rows = int((populated & ~has_ticker).sum())

        usable = body.loc[has_ticker].copy()
        # The file carries no market cap; the importer leaves the stored value alone.
        usable["Market Cap (USD)"] = pd.NA
        usable = usable.drop_duplicates(subset=["Ticker"], keep="last")
        return usable[[*FIXED_COLUMNS, variable_name]].reset_index(drop=True), skipped_rows
