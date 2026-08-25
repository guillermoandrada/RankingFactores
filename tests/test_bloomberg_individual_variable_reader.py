"""Tests for the Bloomberg individual-variable reader."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from modules.infrastructure.ingestion.readers.bloomberg_individual_variable import (
    BloombergIndividualVariableReader,
)


def _workbook(rows: list[list], sheet_name: str = "Volatility 12m") -> bytes:
    buffer = io.BytesIO()
    pd.DataFrame(rows).to_excel(buffer, sheet_name=sheet_name, header=False, index=False)
    return buffer.getvalue()


def _two_period_rows() -> list[list]:
    return [
        ["SPX Index", pd.Timestamp("2026-06-30"), None, None, None,
         "SPX Index", pd.Timestamp("2026-03-30"), None, None, None],
        ["#NAME?", None, None, None, None,
         "#NAME?", None, None, None, None],
        ["AAPL US Equity", "Apple Inc", "Information Technology", "Hardware", 0.25,
         "AAPL US Equity", "Apple Inc", "Information Technology", "Hardware", 0.31],
        ["A UN Equity", "Agilent Technologies", "Health Care", "Health Care Equipment", 0.33,
         "A UN Equity", "Agilent Technologies", "Health Care", "Health Care Equipment", 0.34],
    ]


def test_read_returns_one_import_ready_frame_per_period() -> None:
    result = BloombergIndividualVariableReader().read(_workbook(_two_period_rows()))

    assert result.variable_name == "Volatility 12m"
    assert sorted(result.frames) == ["2026/03/30", "2026/06/30"]

    frame = result.frames["2026/06/30"]
    assert list(frame.columns) == [
        "Ticker",
        "Long Name",
        "GICS Sector Name",
        "GICS Industry Group Name",
        "Market Cap (USD)",
        "Volatility 12m",
    ]
    assert frame["Ticker"].tolist() == ["AAPL", "A"]
    assert frame["Long Name"].tolist() == ["Apple Inc", "Agilent Technologies"]
    assert frame["GICS Sector Name"].tolist() == ["Information Technology", "Health Care"]
    assert frame["Volatility 12m"].tolist() == [0.25, 0.33]
    assert result.skipped_rows == 0


def test_read_keeps_the_index_name_of_each_period() -> None:
    result = BloombergIndividualVariableReader().read(_workbook(_two_period_rows()))

    assert result.index_codes == {
        "2026/06/30": "SPX Index",
        "2026/03/30": "SPX Index",
    }


def test_read_parses_string_dates_into_the_stored_period_format() -> None:
    content = _workbook(
        [
            ["SPX Index", "12/31/2025", None, None, None],
            ["#NAME?", None, None, None, None],
            ["A UN Equity", "Agilent", "Health Care", "Equipment", 0.3],
        ]
    )

    result = BloombergIndividualVariableReader().read(content)

    assert list(result.frames) == ["2025/12/31"]


def test_read_keeps_the_vendor_spelling_of_the_ticker() -> None:
    """Upper-casing here would split 'BFb' and 'GOOG' off their stored securities."""
    content = _workbook(
        [
            ["SPX Index", pd.Timestamp("2025-06-30"), None, None, None],
            [None, None, None, None, None],
            ["BFb US Equity", "Brown-Forman", "Consumer Staples", "Beverages", 0.2],
            ["GOOG UW Equity", "Alphabet Inc", "Communication", "Media", 0.3],
            ["2677689D US Equity", "Delisted Co", "Industrials", "Machinery", 0.4],
        ]
    )

    result = BloombergIndividualVariableReader().read(content)

    assert result.frames["2025/06/30"]["Ticker"].tolist() == ["BFb", "GOOG", "2677689D"]


def test_read_keeps_a_security_whose_value_is_not_numeric() -> None:
    """The security still belongs to the period; the value lands as NULL."""
    content = _workbook(
        [
            ["SPX Index", pd.Timestamp("2026-06-30"), None, None, None],
            [None, None, None, None, None],
            ["A UN Equity", "Agilent", "Health Care", "Equipment", 0.33],
            ["ABT UN Equity", "Abbott", "Health Care", "Equipment", "#N/A N/A"],
        ]
    )

    result = BloombergIndividualVariableReader().read(content)

    frame = result.frames["2026/06/30"]
    assert frame["Ticker"].tolist() == ["A", "ABT"]
    assert pd.isna(frame["Volatility 12m"].iloc[1])
    assert result.skipped_rows == 0


def test_read_counts_rows_dropped_for_a_missing_ticker() -> None:
    content = _workbook(
        [
            ["SPX Index", pd.Timestamp("2026-06-30"), None, None, None],
            [None, None, None, None, None],
            ["A UN Equity", "Agilent", "Health Care", "Equipment", 0.33],
            ["   ", "Orphan row", None, None, 0.44],
        ]
    )

    result = BloombergIndividualVariableReader().read(content)

    assert result.frames["2026/06/30"]["Ticker"].tolist() == ["A"]
    assert result.skipped_rows == 1


def test_read_reports_period_blocks_with_an_unreadable_date() -> None:
    rows = _two_period_rows()
    rows[0][6] = "not a date"

    result = BloombergIndividualVariableReader().read(_workbook(rows))

    assert list(result.frames) == ["2026/06/30"]
    assert result.skipped_periods == ["not a date"]


def test_read_collapses_repeated_tickers_within_a_period() -> None:
    content = _workbook(
        [
            ["SPX Index", pd.Timestamp("2026-06-30"), None, None, None],
            [None, None, None, None, None],
            ["A UN Equity", "Agilent", "Health Care", "Equipment", 0.33],
            ["A UN Equity", "Agilent", "Health Care", "Equipment", 0.44],
        ]
    )

    result = BloombergIndividualVariableReader().read(content)

    frame = result.frames["2026/06/30"]
    assert len(frame) == 1
    assert frame["Volatility 12m"].iloc[0] == 0.44


def test_read_treats_blank_header_cells_as_empty() -> None:
    """An empty cell must not reach the caller as the text 'nan'."""
    rows = _two_period_rows()
    rows[0][0] = None
    rows[0][6] = None

    result = BloombergIndividualVariableReader().read(_workbook(rows))

    assert list(result.frames) == ["2026/06/30"]
    assert result.index_codes == {}
    assert result.skipped_periods == ["column 7"]


def test_read_returns_empty_result_when_no_period_block_is_present() -> None:
    content = _workbook(
        [
            ["SPX Index", "no date here", None, None, None],
            [None, None, None, None, None],
            ["A UN Equity", "Agilent", "Health Care", "Equipment", 0.33],
        ]
    )

    result = BloombergIndividualVariableReader().read(content)

    assert result.is_empty
    assert result.skipped_periods == ["no date here"]


def test_read_uses_the_requested_sheet_as_variable_name() -> None:
    buffer = io.BytesIO()
    rows = pd.DataFrame(
        [
            ["SPX Index", pd.Timestamp("2026-06-30"), None, None, None],
            [None, None, None, None, None],
            ["A UN Equity", "Agilent", "Health Care", "Equipment", 1.1],
        ]
    )
    with pd.ExcelWriter(buffer) as writer:
        rows.to_excel(writer, sheet_name="Volatility 12m", header=False, index=False)
        rows.to_excel(writer, sheet_name="Beta 12m", header=False, index=False)

    result = BloombergIndividualVariableReader().read(buffer.getvalue(), sheet_name="Beta 12m")

    assert result.variable_name == "Beta 12m"
    assert "Beta 12m" in result.frames["2026/06/30"].columns


def test_read_rejects_an_unknown_sheet() -> None:
    with pytest.raises(ValueError, match="not found"):
        BloombergIndividualVariableReader().read(
            _workbook(_two_period_rows()), sheet_name="Missing"
        )


def test_read_rejects_a_sheet_named_after_a_fixed_column() -> None:
    with pytest.raises(ValueError, match="reserved column name"):
        BloombergIndividualVariableReader().read(
            _workbook(_two_period_rows(), sheet_name="Long Name")
        )

def _values_only_rows() -> list[list]:
    """The narrow layout: ticker and value only, two columns per period."""
    return [
        ["SPX Index", pd.Timestamp("2026-06-30"), "SPX Index", pd.Timestamp("2026-03-30")],
        ["ID", "volatility(calendar)", "ID", "volatility(calendar)"],
        ["A UN Equity", 0.334748, "A UN Equity", 0.318650],
        ["AAPL UW Equity", 0.237302, "AAPL UW Equity", 0.316028],
    ]


def test_read_detects_the_ticker_and_value_layout() -> None:
    result = BloombergIndividualVariableReader().read(_workbook(_values_only_rows()))

    assert result.creates_securities is False
    assert sorted(result.frames) == ["2026/03/30", "2026/06/30"]

    frame = result.frames["2026/06/30"]
    assert frame["Ticker"].tolist() == ["A", "AAPL"]
    assert frame["Volatility 12m"].tolist() == [0.334748, 0.237302]


def test_narrow_layout_leaves_the_descriptive_columns_blank() -> None:
    """Blank, not absent: the frame still has to be shaped like any other import."""
    result = BloombergIndividualVariableReader().read(_workbook(_values_only_rows()))

    frame = result.frames["2026/06/30"]
    assert list(frame.columns) == [
        "Ticker",
        "Long Name",
        "GICS Sector Name",
        "GICS Industry Group Name",
        "Market Cap (USD)",
        "Volatility 12m",
    ]
    assert frame["Long Name"].tolist() == ["", ""]
    assert frame["GICS Sector Name"].tolist() == ["", ""]
    assert frame["GICS Industry Group Name"].tolist() == ["", ""]


def test_wide_layout_still_reports_that_it_can_create_securities() -> None:
    result = BloombergIndividualVariableReader().read(_workbook(_two_period_rows()))

    assert result.creates_securities is True


def test_read_detects_a_lone_narrow_block_from_its_values() -> None:
    """With no second block to measure against, the numeric second column decides."""
    content = _workbook(
        [
            ["SPX Index", pd.Timestamp("2026-06-30")],
            ["ID", "volatility"],
            ["A UN Equity", 0.33],
        ]
    )

    result = BloombergIndividualVariableReader().read(content)

    assert result.creates_securities is False
    assert result.frames["2026/06/30"]["Volatility 12m"].tolist() == [0.33]


def test_read_detects_a_lone_wide_block_from_its_name_column() -> None:
    content = _workbook(
        [
            ["SPX Index", pd.Timestamp("2026-06-30"), None, None, None],
            ["#NAME?", None, None, None, None],
            ["A UN Equity", "Agilent", "Health Care", "Equipment", 0.33],
        ]
    )

    result = BloombergIndividualVariableReader().read(content)

    assert result.creates_securities is True
    assert result.frames["2026/06/30"]["Long Name"].tolist() == ["Agilent"]


def test_read_rejects_mixed_layouts() -> None:
    """A trailing block of the other layout would be read out of the wrong columns."""
    rows = [
        ["SPX Index", pd.Timestamp("2026-06-30"),
         "SPX Index", pd.Timestamp("2026-03-30"), None, None, None],
        [None] * 7,
        ["A UN Equity", 0.33,
         "A UN Equity", "Agilent", "Health Care", "Equipment", 0.31],
    ]

    with pytest.raises(ValueError, match="keeps its values in sheet column 7"):
        BloombergIndividualVariableReader().read(_workbook(rows))


def test_read_rejects_inconsistent_block_spacing() -> None:
    rows = [
        ["SPX Index", pd.Timestamp("2026-06-30"), None,
         "SPX Index", pd.Timestamp("2026-03-30"), None, None, None,
         "SPX Index", pd.Timestamp("2025-12-30")],
        [None] * 10,
        ["A UN Equity", 0.33, None, "A UN Equity", 0.31, None, None, None,
         "A UN Equity", 0.30],
    ]

    with pytest.raises(ValueError, match="inconsistent widths"):
        BloombergIndividualVariableReader().read(_workbook(rows))


def test_read_reports_a_period_block_the_sheet_cuts_short() -> None:
    """Truncated blocks are named, not dropped in silence."""
    rows = [
        ["SPX Index", pd.Timestamp("2026-06-30"), None, None, None,
         "SPX Index", pd.Timestamp("2026-03-30")],
        [None] * 7,
        ["A UN Equity", "Agilent", "Health Care", "Equipment", 0.33,
         "A UN Equity", "Agilent"],
    ]

    result = BloombergIndividualVariableReader().read(_workbook(rows))

    assert list(result.frames) == ["2026/06/30"]
    assert result.skipped_periods == ["2026/03/30 (only 2 column(s) left)"]


def test_read_rejects_a_block_width_it_cannot_map() -> None:
    """A spacer column between periods must fail loudly, not read the wrong column."""
    rows = [
        ["SPX Index", pd.Timestamp("2026-06-30"), None,
         "SPX Index", pd.Timestamp("2026-03-30"), None],
        [None, None, None, None, None, None],
        ["A UN Equity", 0.33, None, "A UN Equity", 0.31, None],
    ]

    with pytest.raises(ValueError, match="neither 2 .* nor 5"):
        BloombergIndividualVariableReader().read(_workbook(rows))


def test_narrow_layout_counts_rows_without_a_ticker() -> None:
    content = _workbook(
        [
            ["SPX Index", pd.Timestamp("2026-06-30")],
            ["ID", "volatility"],
            ["A UN Equity", 0.33],
            [None, 0.44],
        ]
    )

    result = BloombergIndividualVariableReader().read(content)

    assert result.frames["2026/06/30"]["Ticker"].tolist() == ["A"]
    assert result.skipped_rows == 1


def test_narrow_layout_reports_an_unreadable_date() -> None:
    rows = _values_only_rows()
    rows[0][3] = "not a date"

    result = BloombergIndividualVariableReader().read(_workbook(rows))

    assert list(result.frames) == ["2026/06/30"]
    assert result.skipped_periods == ["not a date"]
