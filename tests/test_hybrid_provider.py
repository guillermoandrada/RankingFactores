"""Tests for the DB-first / yfinance-fallback price provider."""

from __future__ import annotations

import pandas as pd
import pytest

from modules.infrastructure.db import FinancialDatabase
from modules.infrastructure.market_data.providers.base import PriceMatrixResult
from modules.infrastructure.market_data.providers.hybrid_provider import HybridPriceProvider


class FakeYFinanceProvider:
    """Records what it was asked for and serves a fixed daily series per ticker."""

    provider_name = "yfinance"

    def __init__(self, series_by_ticker: dict[str, pd.Series] | None = None) -> None:
        self.series_by_ticker = series_by_ticker or {}
        self.requested: list[str] = []
        self.frequencies: list[str] = []

    def fetch_price_matrix(
        self,
        identifiers: list[str],
        *,
        start_date: str,
        end_date: str,
        frequency: str = "daily",
    ) -> PriceMatrixResult:
        _ = (start_date, end_date)
        self.requested.extend(identifiers)
        self.frequencies.append(frequency)
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


# ── Write-through caching ─────────────────────────────────────────────────────
def test_downloaded_series_is_persisted_and_served_from_the_db_next_time(
    db: FinancialDatabase,
) -> None:
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    fake = FakeYFinanceProvider({"MSFT": _daily(dates, [10.0, 11.0, 12.0])})
    provider = HybridPriceProvider(db=db, yf_provider=fake)

    first = provider.fetch_price_matrix(
        ["MSFT"], start_date="2024-01-01", end_date="2024-01-03"
    )
    fake.requested.clear()
    second = provider.fetch_price_matrix(
        ["MSFT"], start_date="2024-01-01", end_date="2024-01-03"
    )

    assert first.prices["MSFT"].tolist() == [10.0, 11.0, 12.0]
    assert second.prices["MSFT"].tolist() == [10.0, 11.0, 12.0]
    assert fake.requested == []
    cached = db.list_cached_tickers()
    assert cached[0]["ticker"] == "MSFT"
    assert cached[0]["row_count"] == 3
    assert cached[0]["sources"] == ["yfinance"]


def test_persisted_rows_are_keyed_canonically(db: FinancialDatabase) -> None:
    """A download requested as 'aapl us equity' must be reusable as 'AAPL'."""
    fake = FakeYFinanceProvider(
        {"aapl us equity": _daily(["2024-01-01", "2024-01-03"], [100.0, 102.0])}
    )
    provider = HybridPriceProvider(db=db, yf_provider=fake)

    provider.fetch_price_matrix(
        ["aapl us equity"], start_date="2024-01-01", end_date="2024-01-03"
    )

    matrix = db.query_price_matrix(["AAPL"], start_date="2024-01-01", end_date="2024-01-03")
    assert matrix["AAPL"].tolist() == [100.0, 102.0]


def test_persisting_never_overwrites_manually_uploaded_prices(
    db: FinancialDatabase,
) -> None:
    _store(db, "AAPL", ["2024-01-01"], [999.0])
    fake = FakeYFinanceProvider(
        {"AAPL": _daily(["2024-01-01", "2024-01-02", "2024-01-03"], [1.0, 2.0, 3.0])}
    )
    provider = HybridPriceProvider(db=db, yf_provider=fake)

    provider.fetch_price_matrix(["AAPL"], start_date="2024-01-01", end_date="2024-01-03")

    matrix = db.query_price_matrix(["AAPL"], start_date="2024-01-01", end_date="2024-01-03")
    assert matrix["AAPL"].tolist() == [999.0, 2.0, 3.0]
    assert db.list_cached_tickers()[0]["sources"] == ["bloomberg", "yfinance"]


def test_monthly_requests_download_daily_so_the_cache_stays_daily(
    db: FinancialDatabase,
) -> None:
    """
    Month-end resampling stamps a bar on the calendar month end, not the last
    trading day, so monthly bars must never reach the daily price table.
    """
    fake = FakeYFinanceProvider(
        {
            "MSFT": _daily(
                ["2024-01-30", "2024-01-31", "2024-02-28", "2024-02-29"],
                [10.0, 11.0, 20.0, 22.0],
            )
        }
    )
    provider = HybridPriceProvider(db=db, yf_provider=fake)

    result = provider.fetch_price_matrix(
        ["MSFT"],
        start_date="2024-01-01",
        end_date="2024-02-29",
        frequency="monthly",
    )

    assert fake.frequencies == ["daily"]
    assert result.prices["MSFT"].tolist() == [11.0, 22.0]
    assert db.list_cached_tickers()[0]["row_count"] == 4


def test_a_full_daily_cache_serves_a_monthly_request_without_downloading(
    db: FinancialDatabase,
) -> None:
    """Month-end bounds, or every monthly request re-downloads a complete cache."""
    _store(
        db,
        "AAPL",
        ["2024-01-15", "2024-01-31", "2024-02-15", "2024-02-29"],
        [10.0, 11.0, 20.0, 22.0],
    )
    fake = FakeYFinanceProvider()
    provider = HybridPriceProvider(db=db, yf_provider=fake)

    result = provider.fetch_price_matrix(
        ["AAPL"],
        start_date="2024-01-01",
        end_date="2024-02-29",
        frequency="monthly",
    )

    assert fake.requested == []
    assert result.prices["AAPL"].tolist() == [11.0, 22.0]
    assert result.partial_coverage == []


def test_refetching_replaces_the_whole_downloaded_window(db: FinancialDatabase) -> None:
    """A split rescales the entire adjusted history; segments must not be mixed."""
    fake = FakeYFinanceProvider(
        {"MSFT": _daily(["2024-01-01", "2024-01-31"], [100.0, 102.0])}
    )
    provider = HybridPriceProvider(db=db, yf_provider=fake)
    provider.fetch_price_matrix(["MSFT"], start_date="2024-01-01", end_date="2024-01-31")

    # Post-split: the same dates come back on a new adjustment basis, extended
    # past the cached window so the cache no longer covers the request.
    fake.series_by_ticker["MSFT"] = _daily(
        ["2024-01-01", "2024-01-31", "2024-03-01"], [50.0, 51.0, 52.0]
    )
    result = provider.fetch_price_matrix(
        ["MSFT"], start_date="2024-01-01", end_date="2024-03-01"
    )

    assert result.prices["MSFT"].tolist() == [50.0, 51.0, 52.0]
    matrix = db.query_price_matrix(["MSFT"], start_date="2024-01-01", end_date="2024-03-01")
    assert matrix["MSFT"].tolist() == [50.0, 51.0, 52.0]


def test_persistence_can_be_switched_off(db: FinancialDatabase) -> None:
    fake = FakeYFinanceProvider({"MSFT": _daily(["2024-01-01", "2024-01-03"], [1.0, 2.0])})
    provider = HybridPriceProvider(db=db, yf_provider=fake, persist_downloads=False)

    result = provider.fetch_price_matrix(
        ["MSFT"], start_date="2024-01-01", end_date="2024-01-03"
    )

    assert result.prices["MSFT"].tolist() == [1.0, 2.0]
    assert db.list_cached_tickers() == []


def test_a_cache_write_failure_does_not_break_the_read(db: FinancialDatabase) -> None:
    class FailingDatabase:
        def query_price_matrix(self, *_args, **_kwargs) -> pd.DataFrame:
            return pd.DataFrame()

        def replace_price_data_for_source(self, *_args, **_kwargs) -> int:
            raise RuntimeError("disk full")

    fake = FakeYFinanceProvider({"MSFT": _daily(["2024-01-01", "2024-01-03"], [1.0, 2.0])})
    provider = HybridPriceProvider(db=FailingDatabase(), yf_provider=fake)

    result = provider.fetch_price_matrix(
        ["MSFT"], start_date="2024-01-01", end_date="2024-01-03"
    )

    assert result.prices["MSFT"].tolist() == [1.0, 2.0]
