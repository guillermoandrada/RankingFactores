"""Tests for the prices router."""

from __future__ import annotations

import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.dependencies import get_price_service


class FakePriceService:
    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.latest_requested: list[str] = []

    def ingest_from_file(self, file_content: bytes, filename: str) -> dict:
        if not file_content:
            raise ValueError(f"No valid price rows found in '{filename}'.")
        return {
            "tickers_imported": ["AAPL"],
            "tickers_skipped": [],
            "rows_written": 2,
            "rows_skipped": 0,
            "date_range": {"min": "2024-01-02", "max": "2024-01-03"},
        }

    def list_cached_tickers(self) -> list[dict]:
        return [
            {
                "ticker": "AAPL",
                "min_date": "2024-01-02",
                "max_date": "2024-01-03",
                "row_count": 2,
            }
        ]

    def delete_tickers(self, tickers: list[str]) -> dict:
        cleaned = [t.strip().upper() for t in tickers if t.strip()]
        if not cleaned:
            raise ValueError("No valid tickers provided.")
        self.deleted.extend(cleaned)
        return {"deleted_rows": len(cleaned), "tickers": cleaned}

    def get_latest_closes(self, tickers: list[str]) -> dict:
        self.latest_requested.extend(tickers)
        return {
            "closes": {"AAPL": 190.5},
            "resolved_identifiers": {"AAPL": "AAPL"},
            "missing_identifiers": [t for t in tickers if t != "AAPL"],
        }


@pytest.fixture(name="fake_service")
def _fake_service() -> FakePriceService:
    service = FakePriceService()
    api_main.app.dependency_overrides[get_price_service] = lambda: service
    yield service
    api_main.app.dependency_overrides.clear()


@pytest.fixture(name="client")
def _client(fake_service: FakePriceService) -> TestClient:
    _ = fake_service
    return TestClient(api_main.app)


def _xlsx_bytes() -> bytes:
    buffer = io.BytesIO()
    pd.DataFrame([["Date", "AAPL"], ["2024-01-02", 100.0]]).to_excel(
        buffer, header=False, index=False
    )
    return buffer.getvalue()


def test_upload_returns_201_with_import_summary(client: TestClient) -> None:
    response = client.post(
        "/prices/upload",
        files={"file": ("prices.xlsx", _xlsx_bytes())},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["tickers_imported"] == ["AAPL"]
    assert payload["rows_written"] == 2
    assert payload["rows_skipped"] == 0


def test_upload_returns_422_when_nothing_parses(client: TestClient) -> None:
    response = client.post("/prices/upload", files={"file": ("prices.xlsx", b"")})

    assert response.status_code == 422
    assert "No valid price rows" in response.json()["detail"]


def test_list_cached_tickers(client: TestClient) -> None:
    response = client.get("/prices/tickers")

    assert response.status_code == 200
    assert response.json()["tickers"][0]["ticker"] == "AAPL"


def test_delete_tickers(client: TestClient, fake_service: FakePriceService) -> None:
    response = client.request("DELETE", "/prices/tickers", json=["aapl"])

    assert response.status_code == 200
    assert response.json() == {"deleted_rows": 1, "tickers": ["AAPL"]}
    assert fake_service.deleted == ["AAPL"]


def test_delete_rejects_an_empty_list(client: TestClient) -> None:
    response = client.request("DELETE", "/prices/tickers", json=[])

    assert response.status_code == 400


def test_delete_returns_422_when_no_ticker_is_usable(client: TestClient) -> None:
    response = client.request("DELETE", "/prices/tickers", json=["  "])

    assert response.status_code == 422


def test_latest_endpoint_splits_comma_separated_tickers(
    client: TestClient,
    fake_service: FakePriceService,
) -> None:
    response = client.get("/prices/latest", params={"tickers": "AAPL, MSFT"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["closes"] == {"AAPL": 190.5}
    assert payload["missing_identifiers"] == ["MSFT"]
    assert fake_service.latest_requested == ["AAPL", "MSFT"]


def test_latest_endpoint_rejects_a_blank_ticker_list(client: TestClient) -> None:
    response = client.get("/prices/latest", params={"tickers": " , "})

    assert response.status_code == 400


def test_latest_route_is_not_shadowed_by_the_ticker_listing(client: TestClient) -> None:
    """'/prices/latest' must resolve to the latest-closes handler, not the listing."""
    response = client.get("/prices/latest", params={"tickers": "AAPL"})

    assert "closes" in response.json()
