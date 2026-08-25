"""Tests for the Bloomberg wide-format close-price reader."""

from __future__ import annotations

import io

import pandas as pd

from modules.infrastructure.ingestion.readers.bloomberg_prices import BloombergPriceFileReader


def _workbook(rows: list[list]) -> bytes:
    buffer = io.BytesIO()
    pd.DataFrame(rows).to_excel(buffer, header=False, index=False)
    return buffer.getvalue()


def test_read_parses_wide_layout_into_long_rows() -> None:
    content = _workbook(
        [
            ["Bloomberg export", None, None],
            ["Date", "AAPL", "MSFT"],
            ["2024-01-02", 100.0, 200.0],
            ["2024-01-03", 101.0, 201.0],
        ]
    )

    result = BloombergPriceFileReader().read(content)

    assert list(result.frame.columns) == ["ticker", "price_date", "close_price", "source"]
    assert len(result.frame) == 4
    assert set(result.frame["ticker"]) == {"AAPL", "MSFT"}
    assert set(result.frame["source"]) == {"bloomberg"}
    assert result.frame["price_date"].min() == "2024-01-02"
    assert result.skipped_rows == 0


def test_read_canonicalizes_bloomberg_headers() -> None:
    """The header form must not decide whether the cache is ever readable."""
    content = _workbook(
        [
            [None, None, None],
            ["Date", "AAPL US Equity", "brk/b UN Equity"],
            ["2024-01-02", 100.0, 400.0],
        ]
    )

    result = BloombergPriceFileReader().read(content)

    assert sorted(result.frame["ticker"].unique()) == ["AAPL", "BRK/B"]


def test_read_reports_unusable_headers_instead_of_dropping_them_silently() -> None:
    content = _workbook(
        [
            [None, None, None],
            ["Date", "AAPL", "   "],
            ["2024-01-02", 100.0, 5.0],
        ]
    )

    result = BloombergPriceFileReader().read(content)

    assert sorted(result.frame["ticker"].unique()) == ["AAPL"]
    assert result.frame["close_price"].tolist() == [100.0]


def test_read_counts_rows_dropped_for_bad_dates_and_prices() -> None:
    content = _workbook(
        [
            [None, None],
            ["Date", "AAPL"],
            ["2024-01-02", 100.0],
            ["not a date", 101.0],
            ["2024-01-04", "n/a"],
        ]
    )

    result = BloombergPriceFileReader().read(content)

    assert result.frame["price_date"].tolist() == ["2024-01-02"]
    assert result.skipped_rows == 2


def test_read_handles_excel_serial_dates() -> None:
    content = _workbook(
        [
            [None, None],
            ["Date", "AAPL"],
            [pd.Timestamp("2024-03-15"), 175.5],
        ]
    )

    result = BloombergPriceFileReader().read(content)

    assert result.frame["price_date"].tolist() == ["2024-03-15"]


def test_read_collapses_duplicate_headers_for_one_security() -> None:
    content = _workbook(
        [
            [None, None, None],
            ["Date", "AAPL US Equity", "AAPL"],
            ["2024-01-02", 100.0, 111.0],
        ]
    )

    result = BloombergPriceFileReader().read(content)

    assert len(result.frame) == 1
    assert result.frame.iloc[0]["close_price"] == 111.0


def test_read_returns_empty_result_for_headerless_file() -> None:
    content = _workbook([["only"], ["two"], ["rows"]])

    result = BloombergPriceFileReader().read(content)

    assert result.is_empty
    assert list(result.frame.columns) == ["ticker", "price_date", "close_price", "source"]
