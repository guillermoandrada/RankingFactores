"""Hybrid price provider: DB-cached prices first, yfinance fallback."""

from __future__ import annotations

import logging

import pandas as pd

from modules.infrastructure.db import FinancialDatabase
from modules.infrastructure.market_data.providers.base import (
    BasePriceProvider,
    PriceMatrixResult,
    to_period_end,
)
from modules.infrastructure.market_data.providers.yfinance_provider import YFinancePriceProvider
from modules.shared.tickers import canonical_ticker_map


class HybridPriceProvider(BasePriceProvider):
    """
    Check the local price_data DB table first; fall back to yfinance for any
    ticker whose cached history does not cover the requested date range.

    Manually uploaded Bloomberg prices take priority over yfinance — this allows
    the user to supply correct prices for delisted or unlisted securities. The
    priority applies per observation, not per ticker: a partially cached security
    keeps its cached values and has the remaining dates filled from yfinance,
    rather than silently returning a truncated series.
    """

    _DEFAULT_COVERAGE_TOLERANCE_DAYS = 7

    def __init__(
        self,
        db: FinancialDatabase,
        yf_provider: YFinancePriceProvider,
        coverage_tolerance_days: int = _DEFAULT_COVERAGE_TOLERANCE_DAYS,
    ) -> None:
        self._db = db
        self._yf = yf_provider
        self._coverage_tolerance = pd.Timedelta(days=int(coverage_tolerance_days))
        self._logger = logging.getLogger(__name__)

    @property
    def provider_name(self) -> str:
        return "hybrid"

    def fetch_price_matrix(
        self,
        identifiers: list[str],
        *,
        start_date: str,
        end_date: str,
        frequency: str = "daily",
    ) -> PriceMatrixResult:
        canonical_by_identifier = canonical_ticker_map(list(identifiers))
        if not canonical_by_identifier:
            return PriceMatrixResult(prices=pd.DataFrame())

        cached_series = self._load_cached_series(
            canonical_by_identifier,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
        )
        fallback_needed = [
            identifier
            for identifier in canonical_by_identifier
            if not self._covers_window(
                cached_series.get(identifier), start_date=start_date, end_date=end_date
            )
        ]
        fallback = self._fetch_fallback(
            fallback_needed,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
        )

        series_by_identifier: dict[str, pd.Series] = {}
        resolved: dict[str, str] = {}
        missing: list[str] = []
        partial: list[str] = []

        for identifier, canonical in canonical_by_identifier.items():
            cached = cached_series.get(identifier)
            fetched = fallback.prices.get(identifier) if not fallback.prices.empty else None
            if fetched is not None:
                # Idempotent: providers already honour the convention, but the
                # composition must not depend on that to keep one shared index.
                fetched = to_period_end(fetched.dropna(), frequency)

            if cached is not None and fetched is not None:
                # Cached observations win; yfinance only fills the gaps.
                series = cached.combine_first(fetched).dropna()
                resolved[identifier] = canonical
            elif cached is not None:
                series = cached
                resolved[identifier] = canonical
            elif fetched is not None:
                series = fetched.dropna()
                resolved[identifier] = fallback.resolved_identifiers.get(identifier, identifier)
            else:
                missing.append(identifier)
                continue

            if series.empty:
                missing.append(identifier)
                continue
            if not self._covers_window(series, start_date=start_date, end_date=end_date):
                partial.append(identifier)
            series_by_identifier[identifier] = series.rename(identifier)

        if not series_by_identifier:
            return PriceMatrixResult(
                prices=pd.DataFrame(),
                resolved_identifiers=resolved,
                missing_identifiers=sorted(missing),
                partial_coverage=sorted(partial),
            )

        price_matrix = pd.concat(series_by_identifier.values(), axis=1).sort_index()
        price_matrix = price_matrix.loc[:, ~price_matrix.columns.duplicated()]
        return PriceMatrixResult(
            prices=price_matrix,
            resolved_identifiers=resolved,
            missing_identifiers=sorted(missing),
            partial_coverage=sorted(partial),
        )

    def _load_cached_series(
        self,
        canonical_by_identifier: dict[str, str],
        *,
        start_date: str,
        end_date: str,
        frequency: str,
    ) -> dict[str, pd.Series]:
        """Return the cached series per original identifier, keyed canonically in SQL."""
        db_matrix = self._db.query_price_matrix(
            sorted(set(canonical_by_identifier.values())),
            start_date=start_date,
            end_date=end_date,
        )
        if db_matrix.empty:
            return {}

        cached: dict[str, pd.Series] = {}
        for identifier, canonical in canonical_by_identifier.items():
            if canonical not in db_matrix.columns:
                continue
            series = db_matrix[canonical].dropna()
            if series.empty:
                continue
            cached[identifier] = to_period_end(series, frequency)
        return cached

    def _fetch_fallback(
        self,
        identifiers: list[str],
        *,
        start_date: str,
        end_date: str,
        frequency: str,
    ) -> PriceMatrixResult:
        if not identifiers:
            return PriceMatrixResult(prices=pd.DataFrame())
        return self._yf.fetch_price_matrix(
            identifiers,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
        )

    def _covers_window(
        self,
        series: pd.Series | None,
        *,
        start_date: str,
        end_date: str,
    ) -> bool:
        """
        True when the series spans the requested window within tolerance.

        The tolerance absorbs weekends and holiday closures at either end; without
        it, a window starting on a Saturday would look uncovered for every
        security.
        """
        if series is None or series.empty:
            return False
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        tolerance = self._effective_tolerance(start=start, end=end)
        return series.index[0] <= start + tolerance and series.index[-1] >= end - tolerance

    def _effective_tolerance(self, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.Timedelta:
        """
        Clamp the calendar tolerance to half the requested window.

        A fixed tolerance would let a single cached observation satisfy any window
        shorter than the tolerance itself — the exact case where falling back to
        the secondary source matters most.
        """
        half_window = max(pd.Timedelta(0), (end - start) / 2)
        return min(self._coverage_tolerance, half_window)
