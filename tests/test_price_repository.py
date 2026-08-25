"""Tests for price_data persistence."""

from __future__ import annotations

import pandas as pd
import pytest

from modules.infrastructure.db import FinancialDatabase


def _price_rows(ticker: str, dates: list[str], prices: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [ticker] * len(dates),
            "price_date": dates,
            "close_price": prices,
            "source": ["bloomberg"] * len(dates),
        }
    )


@pytest.fixture(name="db")
def _db(tmp_path) -> FinancialDatabase:
    return FinancialDatabase(db_url=f"sqlite:///{tmp_path}/prices.db")


def test_upsert_and_query_round_trip(db: FinancialDatabase) -> None:
    written = db.upsert_price_data(
        _price_rows("AAPL", ["2024-01-02", "2024-01-03"], [100.0, 101.0])
    )

    matrix = db.query_price_matrix(["AAPL"], start_date="2024-01-01", end_date="2024-01-31")

    assert written == 2
    assert list(matrix.columns) == ["AAPL"]
    assert isinstance(matrix.index, pd.DatetimeIndex)
    assert matrix.loc[pd.Timestamp("2024-01-03"), "AAPL"] == 101.0


def test_upsert_replaces_existing_dates(db: FinancialDatabase) -> None:
    db.upsert_price_data(_price_rows("AAPL", ["2024-01-02"], [100.0]))
    db.upsert_price_data(_price_rows("AAPL", ["2024-01-02"], [123.0]))

    matrix = db.query_price_matrix(["AAPL"], start_date="2024-01-01", end_date="2024-01-31")

    assert len(matrix) == 1
    assert matrix.loc[pd.Timestamp("2024-01-02"), "AAPL"] == 123.0


def test_query_respects_range_boundaries_inclusively(db: FinancialDatabase) -> None:
    db.upsert_price_data(
        _price_rows(
            "AAPL",
            ["2024-01-01", "2024-01-15", "2024-02-01"],
            [10.0, 20.0, 30.0],
        )
    )

    matrix = db.query_price_matrix(["AAPL"], start_date="2024-01-01", end_date="2024-01-15")

    assert matrix["AAPL"].tolist() == [10.0, 20.0]


def test_query_returns_empty_frame_for_unknown_ticker(db: FinancialDatabase) -> None:
    db.upsert_price_data(_price_rows("AAPL", ["2024-01-02"], [100.0]))

    assert db.query_price_matrix(["MSFT"], start_date="2024-01-01", end_date="2024-01-31").empty
    assert db.query_price_matrix([], start_date="2024-01-01", end_date="2024-01-31").empty


def test_upsert_drops_non_numeric_prices(db: FinancialDatabase) -> None:
    frame = _price_rows("AAPL", ["2024-01-02", "2024-01-03"], [100.0, 101.0])
    frame["close_price"] = frame["close_price"].astype(object)
    frame.loc[1, "close_price"] = "n/a"

    written = db.upsert_price_data(frame)

    assert written == 1


def test_upsert_rejects_frames_missing_required_columns(db: FinancialDatabase) -> None:
    with pytest.raises(ValueError):
        db.upsert_price_data(pd.DataFrame({"ticker": ["AAPL"], "close_price": [1.0]}))


def test_list_and_delete_cached_tickers(db: FinancialDatabase) -> None:
    db.upsert_price_data(_price_rows("AAPL", ["2024-01-02", "2024-01-03"], [100.0, 101.0]))
    db.upsert_price_data(_price_rows("BRK/B", ["2024-01-02"], [400.0]))

    listed = db.list_cached_tickers()
    assert [row["ticker"] for row in listed] == ["AAPL", "BRK/B"]
    assert listed[0] == {
        "ticker": "AAPL",
        "min_date": "2024-01-02",
        "max_date": "2024-01-03",
        "row_count": 2,
    }

    deleted = db.delete_price_data_for_tickers(["AAPL"])

    assert deleted == 2
    assert [row["ticker"] for row in db.list_cached_tickers()] == ["BRK/B"]
    assert db.delete_price_data_for_tickers([]) == 0
