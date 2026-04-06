from __future__ import annotations

from fastapi.testclient import TestClient

import api.main as api_main
import api.routers.backtests as backtests_router


def test_backtests_portfolio_endpoint(monkeypatch) -> None:
    class FakeService:
        def backtest_portfolio(self, request):
            return {
                "provider": "fake",
                "methodology": request.methodology,
                "summary": {"ending_value": 1.1},
                "warnings": [],
                "series": [],
            }

    def _fake_service():
        return FakeService()

    monkeypatch.setattr(backtests_router.dependencies, "get_backtest_service", _fake_service)
    client = TestClient(api_main.app)

    response = client.post(
        "/backtests/portfolio",
        json={
            "portfolio": [{"ticker": "AAA", "target_weight": 1.0}],
            "start_date": "2024-01-01",
            "end_date": "2024-01-05",
        },
    )

    assert response.status_code == 200
    assert response.json()["provider"] == "fake"


def test_backtests_strategy_endpoint_rejects_missing_periods(monkeypatch) -> None:
    class FakeDb:
        def list_periods(self):
            return ["2024 Q1"]

    def _fake_db():
        return FakeDb()

    monkeypatch.setattr(backtests_router.dependencies, "get_db", _fake_db)
    client = TestClient(api_main.app)

    response = client.post(
        "/backtests/strategy",
        json={
            "portfolio_request": {
                "scoring_profile": "quality",
                "strategy": "legacy_rebalance",
                "construction_mode": "new_portfolio",
            },
            "schedule": [
                {
                    "period": "2024 Q2",
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-05",
                }
            ],
        },
    )

    assert response.status_code == 404
