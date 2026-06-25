"""Hybrid price provider: DB-cached prices first, yfinance fallback."""

from __future__ import annotations

import logging

import pandas as pd

from modules.infrastructure.db import FinancialDatabase
from modules.infrastructure.market_data.providers.base import BasePriceProvider, PriceMatrixResult
from modules.infrastructure.market_data.providers.yfinance_provider import YFinancePriceProvider


class HybridPriceProvider(BasePriceProvider):
    """
    Check the local price_data DB table first; fall back to yfinance for any
    ticker without DB coverage in the requested date range.

    Manually uploaded Bloomberg prices take priority over yfinance — this allows
    the user to supply correct prices for delisted or unlisted securities.
    """

    def __init__(
        self,
        db: FinancialDatabase,
        yf_provider: YFinancePriceProvider,
        auto_cache_yfinance: bool = False,
    ) -> None:
        self._db = db
        self._yf = yf_provider
        self._auto_cache = auto_cache_yfinance
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
        unique_ids = list(
            dict.fromkeys(
                str(i or "").strip() for i in identifiers if str(i or "").strip()
            )
        )
        if not unique_ids:
            return PriceMatrixResult(prices=pd.DataFrame())

        db_matrix = self._db.query_price_matrix(
            unique_ids, start_date=start_date, end_date=end_date
        )

        db_hit_ids = [
            ident
            for ident in unique_ids
            if ident in db_matrix.columns and db_matrix[ident].notna().any()
        ]
        yf_needed_ids = [ident for ident in unique_ids if ident not in db_hit_ids]

        series_by_id: dict[str, pd.Series] = {}
        resolved: dict[str, str] = {ident: ident for ident in db_hit_ids}
        missing: list[str] = []

        for ident in db_hit_ids:
            series = db_matrix[ident].dropna()
            if frequency == "monthly":
                series = series.resample("ME").last().dropna()
            series_by_id[ident] = series.rename(ident)

        if yf_needed_ids:
            yf_result = self._yf.fetch_price_matrix(
                yf_needed_ids,
                start_date=start_date,
                end_date=end_date,
                frequency=frequency,
            )
            resolved.update(yf_result.resolved_identifiers)
            missing.extend(yf_result.missing_identifiers)
            for ident in yf_needed_ids:
                if ident in yf_result.prices.columns:
                    series_by_id[ident] = yf_result.prices[ident]

        if not series_by_id:
            return PriceMatrixResult(
                prices=pd.DataFrame(),
                resolved_identifiers=resolved,
                missing_identifiers=missing,
            )

        price_matrix = pd.concat(series_by_id.values(), axis=1).sort_index()
        price_matrix = price_matrix.loc[:, ~price_matrix.columns.duplicated()]
        return PriceMatrixResult(
            prices=price_matrix,
            resolved_identifiers=resolved,
            missing_identifiers=missing,
        )
