from __future__ import annotations

from datetime import date

import pandas as pd

from api.schemas.backtests import PortfolioBacktestBody, StrategyBacktestBody
from api.schemas.portfolios import PortfolioBuildBody
from api.services.backtest_service import BacktestService
from modules.infrastructure.market_data.providers.base import PriceMatrixResult


class FakePriceProvider:
    provider_name = "fake"

    def fetch_price_matrix(
        self,
        identifiers: list[str],
        *,
        start_date: str,
        end_date: str,
        frequency: str = "daily",
    ) -> PriceMatrixResult:
        _ = frequency
        index = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
        data: dict[str, list[float]] = {}
        resolved: dict[str, str] = {}
        for identifier in identifiers:
            if identifier == "AAA":
                data[identifier] = [100.0, 110.0, 121.0]
                resolved[identifier] = "AAA"
            elif identifier == "BBB":
                data[identifier] = [50.0, 55.0, 60.5]
                resolved[identifier] = "BBB"
            elif identifier == "SPY":
                data[identifier] = [100.0, 105.0, 110.25]
                resolved[identifier] = "SPY"
        frame = pd.DataFrame(data, index=index)
        frame = frame[(frame.index >= pd.Timestamp(start_date)) & (frame.index <= pd.Timestamp(end_date))]
        missing = [identifier for identifier in identifiers if identifier not in resolved]
        return PriceMatrixResult(
            prices=frame,
            resolved_identifiers=resolved,
            missing_identifiers=missing,
        )


class FakePortfolioService:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def construct_portfolio(self, period: str, request: PortfolioBuildBody) -> dict:
        self.calls.append(period)
        ticker = "AAA" if period == "2024 Q1" else "BBB"
        return {
            "summary": {"position_count": 1},
            "notes": [f"constructed {period} with {request.scoring_profile}"],
            "portfolio": [
                {
                    "ticker": ticker,
                    "target_weight": 1.0,
                    "name": ticker,
                    "sector": "",
                    "industry": "",
                }
            ],
        }


def test_backtest_service_portfolio_backtest_with_benchmark() -> None:
    service = BacktestService(
        portfolio_service=FakePortfolioService(),
        price_provider=FakePriceProvider(),
    )
    request = PortfolioBacktestBody(
        portfolio=[{"ticker": "AAA", "target_weight": 1.0}],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 3),
        methodology="fixed_weights",
        benchmark_ticker="SPY",
        capital_base=100.0,
    )

    result = service.backtest_portfolio(request)

    assert result["provider"] == "fake"
    assert abs(float(result["summary"]["ending_value"]) - 121.0) < 1e-9
    assert result["summary"]["benchmark_total_return"] is not None
    assert result["components"][0]["ticker"] == "AAA"
    assert abs(float(result["components"][0]["total_return"]) - 0.21) < 1e-9
    assert len(result["series"]) == 3


def test_backtest_service_strategy_stitches_period_windows() -> None:
    fake_portfolio_service = FakePortfolioService()
    service = BacktestService(
        portfolio_service=fake_portfolio_service,
        price_provider=FakePriceProvider(),
    )
    request = StrategyBacktestBody(
        portfolio_request=PortfolioBuildBody(
            scoring_profile="quality",
            strategy="legacy_rebalance",
            construction_mode="new_portfolio",
        ),
        schedule=[
            {
                "period": "2024 Q1",
                "start_date": date(2024, 1, 1),
                "end_date": date(2024, 1, 2),
            },
            {
                "period": "2024 Q2",
                "start_date": date(2024, 1, 3),
                "end_date": date(2024, 1, 3),
            },
        ],
        methodology="drifting_weights",
    )

    result = service.backtest_strategy(request)

    assert fake_portfolio_service.calls == ["2024 Q1", "2024 Q2"]
    assert len(result["intervals"]) == 2
    assert result["intervals"][0]["components"][0]["ticker"] == "AAA"
    assert result["intervals"][1]["components"][0]["ticker"] == "BBB"
    assert result["summary"]["ending_value"] is not None
    assert len(result["series"]) == 3


def test_backtest_service_warns_about_partial_price_coverage() -> None:
    """A position priced for only part of the window must not look complete."""

    class PartiallyCoveredProvider(FakePriceProvider):
        def fetch_price_matrix(self, identifiers, *, start_date, end_date, frequency="daily"):
            result = super().fetch_price_matrix(
                identifiers,
                start_date=start_date,
                end_date=end_date,
                frequency=frequency,
            )
            result.partial_coverage = [i for i in identifiers if i == "AAA"]
            return result

    service = BacktestService(
        portfolio_service=FakePortfolioService(),
        price_provider=PartiallyCoveredProvider(),
    )
    request = PortfolioBacktestBody(
        portfolio=[{"ticker": "AAA", "target_weight": 1.0}],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 3),
        capital_base=100.0,
    )

    result = service.backtest_portfolio(request)

    assert any("does not span the full window" in warning for warning in result["warnings"])
    assert any("AAA" in warning for warning in result["warnings"])
