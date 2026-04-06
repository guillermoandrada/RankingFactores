from __future__ import annotations

from fastapi.testclient import TestClient
import pandas as pd
import pytest
from pydantic import ValidationError

import api.main as api_main
import api.routers.portfolios as portfolios_router
import api.services.portfolio_service as portfolio_service_module
from api.schemas.portfolios import PortfolioBuildBody
from api.services.portfolio_service import PortfolioService
from modules.portfolio.long_short import build_long_short_portfolio
from modules.portfolio.legacy_rebalance import build_legacy_rebalance_target
from modules.portfolio.models import SecurityCandidate


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
        constraint_type="industry",
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


def test_portfolio_service_legacy_fully_restricted_returns_cash(monkeypatch) -> None:
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
        strategy="legacy_rebalance",
        constraint_type="sector",
        sector_targets={"Tech": 0.0, "Health": 0.0},
        neutral_position=1.0,
        max_position=1.0,
    )

    result = service.construct_portfolio("2024 Q1", request)

    assert result["summary"]["position_count"] == 1
    assert result["portfolio"] == [
        {
            "ticker": "CASH_USD",
            "name": "Cash",
            "sector": "",
            "industry": "",
            "score": None,
            "current_weight": 0.0,
            "target_weight": 1.0,
            "target_amount": 1.0,
        }
    ]


def test_portfolio_service_legacy_partial_allocation_leaves_residual_cash(monkeypatch) -> None:
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
        strategy="legacy_rebalance",
        constraint_type="sector",
        sector_targets={"Tech": 0.5, "Health": 0.0},
        neutral_position=1.0,
        max_position=1.0,
    )

    result = service.construct_portfolio("2024 Q1", request)

    portfolio_by_ticker = {
        row["ticker"]: row
        for row in result["portfolio"]
    }
    assert portfolio_by_ticker["AAA"]["target_weight"] == 0.5
    assert portfolio_by_ticker["CASH_USD"]["target_weight"] == 0.5
    assert any("Residual weight of 0.5 was assigned to cash." == note for note in result["notes"])


def test_portfolio_service_legacy_explicit_positive_group_is_capped(monkeypatch) -> None:
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
        strategy="legacy_rebalance",
        constraint_type="industry",
        industry_targets={"Software": 0.3, "Biotech": 0.0},
        neutral_position=0.3,
        max_position=0.3,
    )

    result = service.construct_portfolio("2024 Q1", request)

    software_weight = sum(
        float(row["target_weight"])
        for row in result["portfolio"]
        if row["industry"] == "Software"
    )
    cash_weight = next(
        float(row["target_weight"])
        for row in result["portfolio"]
        if row["ticker"] == "CASH_USD"
    )
    assert software_weight == 0.3
    assert cash_weight == 0.7


def test_legacy_rebalance_drops_near_zero_residual_positions() -> None:
    candidates = [
        SecurityCandidate(
            ticker=f"SOFT{i}",
            score=float(20 - i),
            name=f"Software {i}",
            sector="Tech",
            industry="Software",
        )
        for i in range(11)
    ]

    positions, diagnostics = build_legacy_rebalance_target(
        candidates,
        constraint_type="industry",
        industry_targets={"Software": 0.3},
        neutral_position=0.03,
        max_position=0.03,
        score_quantile_cutoff=-1.0,
    )

    assert len(positions) == 10
    assert abs(sum(position.target_weight for position in positions) - 0.3) < 1e-9
    assert all(position.target_weight > 0 for position in positions)
    assert all(position.target_weight >= 0.03 - 1e-9 for position in positions)
    assert diagnostics.constraints == []


def test_portfolio_service_rebalance_to_cash_adds_cash_trade(monkeypatch) -> None:
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
        strategy="legacy_rebalance",
        construction_mode="rebalance_existing",
        constraint_type="sector",
        sector_targets={"Tech": 0.0, "Health": 0.0},
        current_holdings=[
            {"ticker": "AAA", "quantity": 1, "price": 100.0},
        ],
        neutral_position=1.0,
        max_position=1.0,
    )

    result = service.construct_portfolio("2024 Q1", request)

    trade_by_ticker = {
        row["ticker"]: row
        for row in result["trades"]
    }
    assert trade_by_ticker["AAA"]["action"] == "sell"
    assert trade_by_ticker["AAA"]["target_weight"] == 0.0
    assert trade_by_ticker["CASH_USD"]["action"] == "raise_cash"
    assert trade_by_ticker["CASH_USD"]["current_weight"] == 0.0
    assert trade_by_ticker["CASH_USD"]["target_weight"] == 1.0


def test_portfolio_service_smart_beta_empty_universe_returns_cash(monkeypatch) -> None:
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
        ethical_filter_rows=[
            {"ticker": "AAA", "ethical_evaluation": "No"},
            {"ticker": "BBB", "ethical_evaluation": "No"},
            {"ticker": "CCC", "ethical_evaluation": "No"},
        ],
    )

    result = service.construct_portfolio("2024 Q1", request)

    assert result["summary"]["excluded_count"] == 3
    assert result["portfolio"][0]["ticker"] == "CASH_USD"
    assert result["portfolio"][0]["target_weight"] == 1.0


def test_portfolio_service_long_short_empty_universe_returns_cash_only(monkeypatch) -> None:
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
        strategy="long_short",
        ethical_filter_rows=[
            {"ticker": "AAA", "ethical_evaluation": "No"},
            {"ticker": "BBB", "ethical_evaluation": "No"},
            {"ticker": "CCC", "ethical_evaluation": "No"},
        ],
    )

    result = service.construct_portfolio("2024 Q1", request)

    assert result["portfolio"] == [
        {
            "ticker": "CASH_USD",
            "name": "Cash",
            "sector": "",
            "industry": "",
            "score": None,
            "current_weight": 0.0,
            "target_weight": 1.0,
            "target_amount": 1.0,
        }
    ]


def test_build_long_short_portfolio_uses_full_exposure_per_leg() -> None:
    candidates = [
        SecurityCandidate(ticker="AAA", score=4.0, name="AAA"),
        SecurityCandidate(ticker="BBB", score=3.0, name="BBB"),
        SecurityCandidate(ticker="CCC", score=2.0, name="CCC"),
        SecurityCandidate(ticker="DDD", score=1.0, name="DDD"),
    ]

    positions, diagnostics = build_long_short_portfolio(
        candidates,
        bucket_count=2,
        long_bucket_count=1,
        short_bucket_count=1,
        weighting="equal",
        gross_exposure=1.0,
        net_exposure=0.0,
    )

    weight_by_ticker = {position.ticker: position.target_weight for position in positions}
    assert weight_by_ticker["AAA"] == 0.5
    assert weight_by_ticker["BBB"] == 0.5
    assert weight_by_ticker["CCC"] == -0.5
    assert weight_by_ticker["DDD"] == -0.5
    assert "long_exposure=1.0, short_exposure=1.0" in diagnostics.notes[-1]


def test_build_long_short_portfolio_keeps_requested_net_exposure() -> None:
    candidates = [
        SecurityCandidate(ticker="AAA", score=4.0, name="AAA"),
        SecurityCandidate(ticker="BBB", score=3.0, name="BBB"),
        SecurityCandidate(ticker="CCC", score=2.0, name="CCC"),
        SecurityCandidate(ticker="DDD", score=1.0, name="DDD"),
    ]

    positions, _ = build_long_short_portfolio(
        candidates,
        bucket_count=2,
        long_bucket_count=1,
        short_bucket_count=1,
        weighting="equal",
        gross_exposure=1.0,
        net_exposure=0.2,
    )

    total_long = sum(position.target_weight for position in positions if position.target_weight > 0)
    total_short = -sum(position.target_weight for position in positions if position.target_weight < 0)
    net = sum(position.target_weight for position in positions)
    assert total_long == 1.1
    assert total_short == 0.9
    assert abs(net - 0.2) < 1e-9


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


def test_portfolio_schema_rejects_mixed_constraint_types() -> None:
    with pytest.raises(ValidationError, match="constraint_type='sector'"):
        PortfolioBuildBody(
            scoring_profile="quality",
            constraint_type="sector",
            sector_targets={"Tech": 0.5},
            industry_targets={"Software": 0.5},
        )


def test_portfolio_service_legacy_zero_weight_restricts_group(monkeypatch) -> None:
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
        strategy="legacy_rebalance",
        constraint_type="sector",
        sector_targets={"Tech": 0.0},
        neutral_position=1.0,
        max_position=1.0,
    )

    result = service.construct_portfolio("2024 Q1", request)

    assert [item["ticker"] for item in result["portfolio"]] == ["CCC"]
    assert result["constraint_diagnostics"]["industry"] == []
    sector_rows = {
        row["group"]: row
        for row in result["constraint_diagnostics"]["sector"]
    }
    assert sector_rows["Tech"]["target_weight"] == 0.0
    assert sector_rows["Tech"]["specified"] is True
    assert sector_rows["Health"]["target_weight"] is None
    assert sector_rows["Health"]["specified"] is False

