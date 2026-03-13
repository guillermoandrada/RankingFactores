from __future__ import annotations

from fastapi.testclient import TestClient
import pandas as pd

import api.main as api_main
import api.routers.portfolios as portfolios_router
import api.services.portfolio_service as portfolio_service_module
from api.schemas.portfolios import PortfolioBuildBody
from api.services.portfolio_service import PortfolioService


class FakeDB:
    def list_periods(self):
        return ["2024 Q1"]

    def get_security_metadata(self, period: str, tickers: list[str] | None = None):
        _ = period
        all_rows = [
            {"ticker": "AAA", "name": "AAA Corp", "sector": "Tech", "industry": "Software"},
            {"ticker": "BBB", "name": "BBB Corp", "sector": "Tech", "industry": "Software"},
            {"ticker": "CCC", "name": "CCC Corp", "sector": "Health", "industry": "Biotech"},
        ]
        if not tickers:
            return all_rows
        return [row for row in all_rows if row["ticker"] in tickers]


def test_portfolio_service_smart_beta(monkeypatch) -> None:
    def _fake_compute(**_kwargs):
        return pd.DataFrame(
            {
                "Ticker": ["AAA", "BBB", "CCC"],
                "Name": ["AAA Corp", "BBB Corp", "CCC Corp"],
                "Scoring": [3.0, 2.0, 1.0],
            }
        )

    monkeypatch.setattr(portfolio_service_module, "compute_ranking", _fake_compute)
    service = PortfolioService(db=FakeDB())
    request = PortfolioBuildBody(
        scoring_profile="quality",
        strategy="smart_beta",
        smart_beta_max_weight=0.60,
    )

    result = service.construct_portfolio("2024 Q1", request)

    assert result["strategy"] == "smart_beta"
    assert result["summary"]["position_count"] == 3
    assert abs(sum(item["target_weight"] for item in result["portfolio"]) - 1.0) < 1e-6


def test_portfolio_service_rebalance_generates_trades(monkeypatch) -> None:
    def _fake_compute(**_kwargs):
        return pd.DataFrame(
            {
                "Ticker": ["AAA", "BBB"],
                "Name": ["AAA Corp", "BBB Corp"],
                "Scoring": [5.0, 1.0],
            }
        )

    monkeypatch.setattr(portfolio_service_module, "compute_ranking", _fake_compute)
    service = PortfolioService(db=FakeDB())
    request = PortfolioBuildBody(
        scoring_profile="quality",
        strategy="legacy_rebalance",
        construction_mode="rebalance_existing",
        current_holdings=[
            {"ticker": "BBB", "quantity": 10, "price": 10.0},
            {"ticker": "CASH_USD", "quantity": 100.0},
        ],
        industry_targets={"Software": 1.0},
        neutral_position=0.50,
        max_position=0.60,
    )

    result = service.construct_portfolio("2024 Q1", request)

    assert result["construction_mode"] == "rebalance_existing"
    assert result["trades"]
    assert any(trade["ticker"] == "AAA" for trade in result["trades"])


def test_portfolios_endpoint(monkeypatch) -> None:
    class FakeService:
        def construct_portfolio(self, period: str, request: PortfolioBuildBody):
            return {
                "strategy": request.strategy,
                "construction_mode": request.construction_mode,
                "portfolio": [],
                "summary": {"total_capital": 1.0, "position_count": 0, "excluded_count": 0, "trade_count": 0},
                "current_portfolio": [],
                "trades": [],
                "excluded": [],
                "constraint_diagnostics": {"sector": [], "industry": []},
                "notes": [],
                "source_count": 0,
                "input_scope": {"period": period},
            }

    def _fake_service():
        return FakeService()

    def _fake_db():
        return FakeDB()

    monkeypatch.setattr(portfolios_router, "get_portfolio_service", _fake_service)
    monkeypatch.setattr(portfolios_router, "get_db", _fake_db)
    client = TestClient(api_main.app)

    response = client.post(
        "/portfolios/2024%20Q1",
        json={
            "scoring_profile": "quality",
            "strategy": "smart_beta",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["strategy"] == "smart_beta"
    assert "portfolio" in payload

