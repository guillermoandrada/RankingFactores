"""Tests for the Bloomberg block-format total-return reader."""

from __future__ import annotations

import datetime as dt
import io

import pandas as pd

from modules.infrastructure.ingestion.readers.bloomberg_total_returns import (
    BloombergTotalReturnFileReader,
)

_VALUE_HEADER = "total_return(calc_interval=range(2025-03-30,2025-06-30,frq=D))"


def total_return_block(
    end: dt.date, start: dt.date, rows: list[tuple[str, object]]
) -> list[list]:
    return [
        ["SPX Index", end, start, None, None],
        ["ID", "name", "gics_sector_name", "gics_industry_group_name", _VALUE_HEADER],
        *[[identifier, "Name", "Sector", "Industry", value] for identifier, value in rows],
    ]


def side_by_side(*blocks: list[list]) -> list[list]:
    height = max(len(block) for block in blocks)
    padded = [block + [[None] * 5] * (height - len(block)) for block in blocks]
    return [sum((block[row] for block in padded), []) for row in range(height)]


def workbook(sheets: dict[str, list[list]]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        for name, rows in sheets.items():
            pd.DataFrame(rows).to_excel(writer, sheet_name=name, header=False, index=False)
    return buffer.getvalue()


def test_read_parses_every_block_of_every_sheet() -> None:
    content = workbook(
        {
            "3m Total Return": side_by_side(
                total_return_block(
                    dt.date(2025, 6, 30), dt.date(2025, 3, 30), [("AAPL UW Equity", 0.1)]
                ),
                total_return_block(
                    dt.date(2025, 3, 30), dt.date(2024, 12, 30), [("AAPL UW Equity", 0.2)]
                ),
            ),
            "12m Total Return": total_return_block(
                dt.date(2025, 6, 30), dt.date(2024, 6, 30), [("MSFT UW Equity", -0.05)]
            ),
        }
    )

    result = BloombergTotalReturnFileReader().read(content)

    assert list(result.frame.columns) == [
        "ticker",
        "start_date",
        "end_date",
        "horizon_months",
        "total_return",
        "source",
    ]
    assert len(result.frame) == 3
    by_window = result.frame.set_index(["ticker", "start_date"])
    assert by_window.loc[("AAPL", "2025-03-30"), "horizon_months"] == 3
    assert by_window.loc[("AAPL", "2024-12-30"), "total_return"] == 0.2
    assert by_window.loc[("MSFT", "2024-06-30"), "horizon_months"] == 12
    assert set(result.frame["source"]) == {"bloomberg"}


def test_read_ignores_sheets_without_blocks() -> None:
    content = workbook(
        {
            "3m Total Return": total_return_block(
                dt.date(2025, 6, 30), dt.date(2025, 3, 30), [("AAPL UW Equity", 0.1)]
            ),
            "Sheet1": [["total_return(formula helper)"], ["volatility(...)"]],
        }
    )

    result = BloombergTotalReturnFileReader().read(content)

    assert result.frame["ticker"].tolist() == ["AAPL"]


def test_read_counts_rows_dropped_for_non_numeric_returns() -> None:
    content = workbook(
        {
            "3m": total_return_block(
                dt.date(2025, 6, 30),
                dt.date(2025, 3, 30),
                [("AAPL UW Equity", 0.1), ("MSFT UW Equity", "#N/A"), ("   ", 0.3)],
            )
        }
    )

    result = BloombergTotalReturnFileReader().read(content)

    assert result.frame["ticker"].tolist() == ["AAPL"]
    assert result.skipped_rows == 1


def test_read_returns_empty_for_a_price_layout() -> None:
    content = workbook({"prices": [[None, None], ["Date", "AAPL"], ["2024-01-02", 100.0]]})

    result = BloombergTotalReturnFileReader().read(content)

    assert result.is_empty
