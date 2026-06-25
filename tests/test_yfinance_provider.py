from __future__ import annotations

import pandas as pd

from modules.infrastructure.market_data.providers.yfinance_provider import YFinancePriceProvider


def test_yfinance_provider_bulk_fetch_with_single_ticker_fallback(monkeypatch) -> None:
    provider = YFinancePriceProvider()
    index = pd.to_datetime(["2024-01-01", "2024-01-02"])

    def _fake_download(*_args, **_kwargs):
        columns = pd.MultiIndex.from_tuples(
            [
                ("Close", "AAA"),
                ("Close", "BBB"),
            ]
        )
        return pd.DataFrame(
            [
                [100.0, None],
                [101.0, None],
            ],
            index=index,
            columns=columns,
        )

    class _FakeTicker:
        def __init__(self, ticker: str) -> None:
            self._ticker = ticker

        def history(self, **_kwargs):
            if self._ticker == "BBB":
                return pd.DataFrame(
                    {"Close": [200.0, 202.0]},
                    index=index,
                )
            return pd.DataFrame()

    monkeypatch.setattr(
        "modules.infrastructure.market_data.providers.yfinance_provider.yf.download",
        _fake_download,
    )
    monkeypatch.setattr(
        "modules.infrastructure.market_data.providers.yfinance_provider.yf.Ticker",
        _FakeTicker,
    )

    result = provider.fetch_price_matrix(
        ["AAA", "BBB"],
        start_date="2024-01-01",
        end_date="2024-01-02",
    )

    assert list(result.prices.columns) == ["AAA", "BBB"]
    assert result.resolved_identifiers == {"AAA": "AAA", "BBB": "BBB"}
    assert result.missing_identifiers == []
    assert float(result.prices.loc[pd.Timestamp("2024-01-02"), "BBB"]) == 202.0


def test_yfinance_provider_normalizes_repo_tickers() -> None:
    assert YFinancePriceProvider.normalize_identifier(" brk/b ") == "BRK-B"
