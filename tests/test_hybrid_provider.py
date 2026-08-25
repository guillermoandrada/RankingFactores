"""Tests for the DB-first / yfinance-fallback price provider."""

from __future__ import annotations

import pandas as pd
import pytest

from modules.infrastructure.db import FinancialDatabase
from modules.infrastructure.market_data.providers.base import PriceMatrixResult
from modules.infrastructure.market_data.providers.hybrid_provider import HybridPriceProvider


class FakeYFinanceProvider:
    """Records what it was asked for and serves a fixed daily series per ticker."""

    provider_name = "fake-yfinance"

    def __init__(self, series_by_ticker: dict[str, pd.Series] | None = None) -> None:
        self.series_by_ticker = series_by_ticker or {}
        self.requested: list[str] = []

    def fetch_price_matrix(
        self,
        identifiers: list[str],
        *,
        start_date: str,
        end_date: str,
        frequency: str = "daily",
    ) -> PriceMatrixResult:
        _ = (start_date, end_date, frequency)
        self.requested.extend(identifiers)
        columns = {
            identifier: self.series_by_ticker[identifier]
            for identifier in identifiers
            if identifier in self.series_by_ticker
        }
        missing = [i for i in identifiers if i not in columns]
        if not columns:
            return PriceMatrixResult(prices=pd.DataFrame(), missing_identifiers=missing)
        return PriceMatrixResult(
            prices=pd.DataFrame(columns),
            resolved_identifiers={i: f"YF:{i}" for i in columns},
            missing_identifiers=missing,
        )


def _daily(dates: list[str], values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.to_datetime(dates))


def _store(db: FinancialDatabase, ticker: str, dates: list[str], values: list[float]) -> None:
    db.upsert_price_data(
        pd.DataFrame(
            {
                "ticker": [ticker] * len(dates),
                "price_date": dates,
                "close_price": values,
                "source": ["bloomberg"] * len(dates),
            }
        )
    )


@pytest.fixture(name="db")
def _db(tmp_path) -> FinancialDatabase:
    return FinancialDatabase(db_url=f"sqlite:///{tmp_path}/prices.db")


def test_cached_prices_take_priority_over_yfinance(db: FinancialDatabase) -> None:
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    _store(db, "AAPL", dates, [999.0, 999.0, 999.0])
    fake = FakeYFinanceProvider({"AAPL": _daily(dates, [1.0, 1.0, 1.0])})
    provider = HybridPriceProvider(db=db, yf_provider=fake)

    result = provider.fetch_price_matrix(
        ["AAPL"], start_date="2024-01-01", end_date="2024-01-03"
    )

    assert result.prices["AAPL"].tolist() == [999.0, 999.0, 999.0]
    assert result.resolved_identifiers["AAPL"] == "AAPL"
    assert fake.requested == []


def test_uncached_tickers_fall_back_to_yfinance(db: FinancialDatabase) -> None:
    dates = ["2024-01-01", "2024-01-03"]
    fake = FakeYFinanceProvider({"MSFT": _daily(dates, [10.0, 12.0])})
    provider = HybridPriceProvider(db=db, yf_provider=fake)

    result = provider.fetch_price_matrix(
        ["MSFT"], start_date="2024-01-01", end_date="2024-01-03"
    )

    assert result.prices["MSFT"].tolist() == [10.0, 12.0]
    assert result.resolved_identifiers["MSFT"] == "YF:MSFT"
    assert result.missing_identifiers == []
    assert fake.requested == ["MSFT"]


def test_bloomberg_headers_resolve_against_canonical_cache(db: FinancialDatabase) -> None:
    """A caller asking for 'BRK/B' must hit rows stored from a 'BRK/B US Equity' upload."""
    dates = ["2024-01-01", "2024-01-03"]
    _store(db, "BRK/B", dates, [400.0, 402.0])
    fake = FakeYFinanceProvider()
    provider = HybridPriceProvider(db=db, yf_provider=fake)

    result = provider.fetch_price_matrix(
        ["BRK/B"], start_date="2024-01-01", end_date="2024-01-03"
    )

    assert result.prices["BRK/B"].tolist() == [400.0, 402.0]
    assert fake.requested == []


def test_columns_are_keyed_by_the_callers_identifier(db: FinancialDatabase) -> None:
    dates = ["2024-01-01", "2024-01-03"]
    _store(db, "AAPL", dates, [100.0, 102.0])
    provider = HybridPriceProvider(db=db, yf_provider=FakeYFinanceProvider())

    result = provider.fetch_price_matrix(
        ["aapl us equity"], start_date="2024-01-01", end_date="2024-01-03"
    )

    assert list(result.prices.columns) == ["aapl us equity"]
    assert result.resolved_identifiers["aapl us equity"] == "AAPL"


def test_partial_cache_is_filled_from_yfinance_without_losing_cached_values(
    db: FinancialDatabase,
) -> None:
    """One uploaded day must not suppress the fallback for the rest of the window."""
    _store(db, "AAPL", ["2024-01-01"], [999.0])
    fake = FakeYFinanceProvider(
        {"AAPL": _daily(["2024-01-01", "2024-01-02", "2024-01-03"], [1.0, 2.0, 3.0])}
    )
    provider = HybridPriceProvider(db=db, yf_provider=fake)

    result = provider.fetch_price_matrix(
        ["AAPL"], start_date="2024-01-01", end_date="2024-01-03"
    )

    assert fake.requested == ["AAPL"]
    # Cached observation wins on its own date; yfinance fills the remainder.
    assert result.prices["AAPL"].tolist() == [999.0, 2.0, 3.0]
    assert result.partial_coverage == []


def test_partial_coverage_is_reported_when_no_source_spans_the_window(
    db: FinancialDatabase,
) -> None:
    _store(db, "DELISTED", ["2024-01-01", "2024-01-02"], [50.0, 51.0])
    provider = HybridPriceProvider(db=db, yf_provider=FakeYFinanceProvider())

    result = provider.fetch_price_matrix(
        ["DELISTED"], start_date="2024-01-01", end_date="2024-06-30"
    )

    assert result.partial_coverage == ["DELISTED"]
    assert result.missing_identifiers == []
    assert result.prices["DELISTED"].tolist() == [50.0, 51.0]


def test_unpriced_tickers_are_reported_as_missing(db: FinancialDatabase) -> None:
    provider = HybridPriceProvider(db=db, yf_provider=FakeYFinanceProvider())

    result = provider.fetch_price_matrix(
        ["NOPE"], start_date="2024-01-01", end_date="2024-01-03"
    )

    assert result.prices.empty
    assert result.missing_identifiers == ["NOPE"]


def test_coverage_tolerance_absorbs_weekend_gaps(db: FinancialDatabase) -> None:
    """A window opening on a Saturday must not mark every security partial."""
    _store(db, "AAPL", ["2024-01-08", "2024-01-12"], [100.0, 104.0])
    provider = HybridPriceProvider(db=db, yf_provider=FakeYFinanceProvider())

    result = provider.fetch_price_matrix(
        ["AAPL"], start_date="2024-01-06", end_date="2024-01-13"
    )

    assert result.partial_coverage == []
    assert result.missing_identifiers == []


def test_monthly_frequency_puts_both_sources_on_one_month_end_index(
    db: FinancialDatabase,
) -> None:
    """DB series resample to month end; yfinance month-start bars must follow suit."""
    _store(db, "CACHED", ["2024-01-31", "2024-02-29"], [10.0, 11.0])
    fake = FakeYFinanceProvider(
        {"FETCHED": _daily(["2024-01-01", "2024-02-01"], [20.0, 22.0])}
    )
    provider = HybridPriceProvider(db=db, yf_provider=fake)

    result = provider.fetch_price_matrix(
        ["CACHED", "FETCHED"],
        start_date="2024-01-01",
        end_date="2024-02-29",
        frequency="monthly",
    )

    assert list(result.prices.index) == [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")]
    assert not result.prices.isna().any().any()


def test_empty_input_returns_empty_result(db: FinancialDatabase) -> None:
    provider = HybridPriceProvider(db=db, yf_provider=FakeYFinanceProvider())

    result = provider.fetch_price_matrix([" ", None], start_date="2024-01-01", end_date="2024-01-03")

    assert result.prices.empty
    assert result.missing_identifiers == []
