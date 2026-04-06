"""Tests for latest adjusted-close resolution."""

from __future__ import annotations

import pandas as pd

from modules.market_data.latest_closes import fetch_latest_adjusted_closes
from modules.market_data.providers.yfinance_provider import YFinancePriceProvider


def test_fetch_latest_adjusted_closes_uses_last_row(monkeypatch) -> None:
    provider = YFinancePriceProvider()

    def fake_fetch(
        self,
        identifiers: list[str],
        *,
        start_date: str,
        end_date: str,
        frequency: str = "daily",
    ):
        from modules.market_data.providers.base import PriceMatrixResult

        idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        prices = pd.DataFrame(
            {"AAA": [10.0, 11.0, 12.5], "BBB": [100.0, float("nan"), 99.0]},
            index=idx,
        )
        return PriceMatrixResult(
            prices=prices,
            resolved_identifiers={"AAA": "AAA", "BBB": "BBB"},
            missing_identifiers=[],
        )

    monkeypatch.setattr(YFinancePriceProvider, "fetch_price_matrix", fake_fetch)

    result = fetch_latest_adjusted_closes(["AAA", "BBB"], provider=provider)

    assert result.closes["AAA"] == 12.5
    assert result.closes["BBB"] == 99.0
    assert result.missing_identifiers == []


def test_fetch_latest_adjusted_closes_reports_empty_series(monkeypatch) -> None:
    provider = YFinancePriceProvider()

    def fake_fetch(self, identifiers, *, start_date, end_date, frequency="daily"):
        from modules.market_data.providers.base import PriceMatrixResult

        idx = pd.to_datetime(["2024-01-02"])
        prices = pd.DataFrame({"ZZZ": [float("nan")]}, index=idx)
        return PriceMatrixResult(prices=prices, resolved_identifiers={}, missing_identifiers=[])

    monkeypatch.setattr(YFinancePriceProvider, "fetch_price_matrix", fake_fetch)

    result = fetch_latest_adjusted_closes(["ZZZ"], provider=provider)

    assert result.closes == {}
    assert "ZZZ" in result.missing_identifiers
