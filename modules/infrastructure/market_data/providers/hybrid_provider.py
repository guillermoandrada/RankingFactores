"""Hybrid price provider: DB-cached prices first, yfinance write-through fallback."""

from __future__ import annotations

import logging

import pandas as pd

from modules.infrastructure.db import FinancialDatabase
from modules.infrastructure.market_data.providers.base import (
    DAILY_FREQUENCY,
    BasePriceProvider,
    PriceMatrixResult,
    to_period_end,
    to_period_end_bound,
)
from modules.infrastructure.market_data.providers.yfinance_provider import YFinancePriceProvider
from modules.shared.tickers import canonical_ticker_map

# The ``source`` label of auto-cached rows. Callers above the provider layer use
# it to refresh downloaded history without touching manual uploads.
PRICE_SOURCE_YFINANCE = "yfinance"


class HybridPriceProvider(BasePriceProvider):
    """
    Serve prices from the local ``price_data`` table, downloading from yfinance
    only what the cache does not already cover — and persisting whatever it
    downloads, so the same window is served from the database next time.

    Manually uploaded Bloomberg prices take priority over yfinance — this allows
    the user to supply correct prices for delisted or unlisted securities. The
    priority applies per observation, not per ticker: a partially cached security
    keeps its cached values and has the remaining dates filled from yfinance,
    rather than silently returning a truncated series.

    Everything is cached at daily granularity, including the series behind a
    monthly request: month-end resampling stamps a bar on the calendar month end
    rather than the last trading day, so storing monthly bars would put dates in
    the table that never traded and would let a sparse twelve-point series look
    like it covered a daily window. Monthly requests therefore download daily
    closes and resample, which is also what the cached path already does — the
    two sources end up agreeing by construction instead of by correction.
    """

    _DEFAULT_COVERAGE_TOLERANCE_DAYS = 7

    def __init__(
        self,
        db: FinancialDatabase,
        yf_provider: YFinancePriceProvider,
        coverage_tolerance_days: int = _DEFAULT_COVERAGE_TOLERANCE_DAYS,
        persist_downloads: bool = True,
    ) -> None:
        self._db = db
        self._yf = yf_provider
        self._coverage_tolerance = pd.Timedelta(days=int(coverage_tolerance_days))
        self._persist_downloads = persist_downloads
        self._logger = logging.getLogger(__name__)

    @property
    def provider_name(self) -> str:
        return "hybrid"

    @property
    def download_source(self) -> str:
        """The ``source`` label written for rows this provider caches."""
        return self._yf.provider_name

    def fetch_price_matrix(
        self,
        identifiers: list[str],
        *,
        start_date: str,
        end_date: str,
        frequency: str = DAILY_FREQUENCY,
    ) -> PriceMatrixResult:
        canonical_by_identifier = canonical_ticker_map(list(identifiers))
        if not canonical_by_identifier:
            return PriceMatrixResult(prices=pd.DataFrame())

        canonical_tickers = sorted(set(canonical_by_identifier.values()))
        cached_series = self._cached_series_by_identifier(
            self._db.query_price_matrix(
                canonical_tickers, start_date=start_date, end_date=end_date
            ),
            canonical_by_identifier,
            frequency=frequency,
        )
        # The authoritative tier is only consulted to break a tie with a fresh
        # download, so an empty cache does not need the second query at all.
        uploaded_series = (
            self._cached_series_by_identifier(
                self._db.query_price_matrix(
                    canonical_tickers,
                    start_date=start_date,
                    end_date=end_date,
                    exclude_source=self.download_source,
                ),
                canonical_by_identifier,
                frequency=frequency,
            )
            if cached_series
            else {}
        )
        fallback_needed = [
            identifier
            for identifier in canonical_by_identifier
            if not self._covers_window(
                cached_series.get(identifier),
                start_date=start_date,
                end_date=end_date,
                frequency=frequency,
            )
        ]
        downloaded = self._download_daily(
            fallback_needed, start_date=start_date, end_date=end_date
        )
        self._persist(
            downloaded,
            canonical_by_identifier=canonical_by_identifier,
            start_date=start_date,
            end_date=end_date,
        )

        series_by_identifier: dict[str, pd.Series] = {}
        resolved: dict[str, str] = {}
        missing: list[str] = []
        partial: list[str] = []

        for identifier, canonical in canonical_by_identifier.items():
            cached = cached_series.get(identifier)
            fetched = (
                downloaded.prices.get(identifier) if not downloaded.prices.empty else None
            )
            if fetched is not None:
                # Downloads always come back daily; the requested frequency is
                # applied here so both sources share one index.
                fetched = to_period_end(fetched.dropna(), frequency)

            if fetched is not None:
                # A fresh download replaced the previously downloaded rows for
                # this window, so only uploaded observations still outrank it.
                # Letting the whole cache win here would keep a pre-split segment
                # in front of a rescaled one and fabricate a jump in returns.
                uploaded = uploaded_series.get(identifier)
                if uploaded is not None:
                    series = uploaded.combine_first(fetched).dropna()
                    resolved[identifier] = canonical
                else:
                    series = fetched.dropna()
                    resolved[identifier] = downloaded.resolved_identifiers.get(
                        identifier, identifier
                    )
            elif cached is not None:
                series = cached
                resolved[identifier] = canonical
            else:
                missing.append(identifier)
                continue

            if series.empty:
                missing.append(identifier)
                continue
            if not self._covers_window(
                series, start_date=start_date, end_date=end_date, frequency=frequency
            ):
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

    @staticmethod
    def _cached_series_by_identifier(
        cached_matrix: pd.DataFrame,
        canonical_by_identifier: dict[str, str],
        *,
        frequency: str,
    ) -> dict[str, pd.Series]:
        """Re-key the canonically-indexed cache matrix onto the caller's identifiers."""
        if cached_matrix.empty:
            return {}

        cached: dict[str, pd.Series] = {}
        for identifier, canonical in canonical_by_identifier.items():
            if canonical not in cached_matrix.columns:
                continue
            series = cached_matrix[canonical].dropna()
            if series.empty:
                continue
            cached[identifier] = to_period_end(series, frequency)
        return cached

    def _download_daily(
        self,
        identifiers: list[str],
        *,
        start_date: str,
        end_date: str,
    ) -> PriceMatrixResult:
        if not identifiers:
            return PriceMatrixResult(prices=pd.DataFrame())
        return self._yf.fetch_price_matrix(
            identifiers,
            start_date=start_date,
            end_date=end_date,
            frequency=DAILY_FREQUENCY,
        )

    def _persist(
        self,
        downloaded: PriceMatrixResult,
        *,
        canonical_by_identifier: dict[str, str],
        start_date: str,
        end_date: str,
    ) -> int:
        """
        Store a download in the price cache, keyed canonically.

        Cache population must never break the read it rode in on, so a write
        failure is logged and the already-resolved prices are returned anyway.
        """
        if not self._persist_downloads or downloaded.prices.empty:
            return 0

        frame = self._to_long_rows(downloaded.prices, canonical_by_identifier)
        if frame.empty:
            return 0
        try:
            return self._db.replace_price_data_for_source(
                frame,
                source=self.download_source,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._logger.warning(
                "Could not cache %d downloaded price rows: %s", len(frame), exc
            )
            return 0

    def _to_long_rows(
        self,
        prices: pd.DataFrame,
        canonical_by_identifier: dict[str, str],
    ) -> pd.DataFrame:
        """Flatten a wide daily price matrix into ``price_data`` rows."""
        columns = [
            column for column in prices.columns if str(column) in canonical_by_identifier
        ]
        if not columns:
            return pd.DataFrame()

        long_rows = (
            prices[columns]
            .rename(columns={column: canonical_by_identifier[str(column)] for column in columns})
            .rename_axis("price_date")
            .reset_index()
            .melt(id_vars="price_date", var_name="ticker", value_name="close_price")
            .dropna(subset=["close_price"])
        )
        if long_rows.empty:
            return pd.DataFrame()

        long_rows["price_date"] = pd.to_datetime(long_rows["price_date"]).dt.strftime("%Y-%m-%d")
        long_rows["source"] = self.download_source
        return long_rows[["ticker", "price_date", "close_price", "source"]]

    def _covers_window(
        self,
        series: pd.Series | None,
        *,
        start_date: str,
        end_date: str,
        frequency: str = DAILY_FREQUENCY,
    ) -> bool:
        """
        True when the series spans the requested window within tolerance.

        The tolerance absorbs weekends and holiday closures at either end; without
        it, a window starting on a Saturday would look uncovered for every
        security. The bounds are snapped to the index convention of ``frequency``
        first, or a fully cached monthly series would be judged against mid-month
        dates it can never carry and be re-downloaded on every request.
        """
        if series is None or series.empty:
            return False
        start = to_period_end_bound(pd.Timestamp(start_date), frequency)
        end = to_period_end_bound(pd.Timestamp(end_date), frequency)
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
