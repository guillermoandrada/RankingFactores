"""Resolve latest adjusted closes for tickers using the Yahoo Finance provider."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from modules.infrastructure.market_data.providers.yfinance_provider import YFinancePriceProvider


@dataclass
class LatestAdjustedClosesResult:
    """Last available adjusted close per requested identifier."""

    closes: dict[str, float]
    resolved_identifiers: dict[str, str] = field(default_factory=dict)
    missing_identifiers: list[str] = field(default_factory=list)


def fetch_latest_adjusted_closes(
    identifiers: list[str],
    *,
    lookback_calendar_days: int = 45,
    provider: YFinancePriceProvider | None = None,
) -> LatestAdjustedClosesResult:
    """
    Fetch the most recent non-null adjusted daily close for each identifier.

    Uses a short historical window so the last row reflects the latest session
    Yahoo returns for the range.
    """
    prov = provider or YFinancePriceProvider()
    unique = list(dict.fromkeys(str(i or "").strip() for i in identifiers if str(i or "").strip()))
    if not unique:
        return LatestAdjustedClosesResult(closes={})

    end = date.today()
    start = end - timedelta(days=lookback_calendar_days)
    matrix_result = prov.fetch_price_matrix(
        unique,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
    )

    closes: dict[str, float] = {}
    stale_columns: list[str] = []

    if not matrix_result.prices.empty:
        for col in matrix_result.prices.columns:
            col_id = str(col)
            series = matrix_result.prices[col_id].dropna()
            if series.empty:
                stale_columns.append(col_id)
            else:
                closes[col_id] = float(series.iloc[-1])

    missing_set = set(matrix_result.missing_identifiers) | set(stale_columns)
    missing_set.difference_update(closes.keys())
    return LatestAdjustedClosesResult(
        closes=closes,
        resolved_identifiers=dict(matrix_result.resolved_identifiers),
        missing_identifiers=sorted(missing_set),
    )
