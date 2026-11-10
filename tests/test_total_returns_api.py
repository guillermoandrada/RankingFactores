"""Tests for the total-returns router."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.dependencies import get_total_return_service


class FakeTotalReturnService:
    def __init__(self) -> None:
        self.deleted_horizons: list[int | None] = []

    def ingest_from_file(self, file_content: bytes, filename: str) -> dict:
        if not file_content:
            raise ValueError(f"No valid total-return rows found in '{filename}'.")
        return {
            "tickers_imported": ["AAPL"],
            "tickers_skipped": [],
            "rows_written": 1,
            "rows_skipped": 0,
            "horizons_months": [3],
            "windows_imported": 1,
            "date_range": {"min": "2025-03-30", "max": "2025-06-30"},
        }

    def list_windows(self) -> list[dict]:
        return [
            {
                "horizon_months": 3,
                "start_date": "2025-03-30",
                "end_date": "2025-06-30",
                "ticker_count": 1,
            }
        ]

    def delete_windows(self, horizon_months: int | None = None) -> dict:
        if horizon_months is not None and horizon_months <= 0:
            raise ValueError("horizon_months must be a positive number of months.")
        self.deleted_horizons.append(horizon_months)
        return {"deleted_rows": 1, "horizon_months": horizon_months}


@pytest.fixture(name="fake_service")
def _fake_service() -> FakeTotalReturnService:
    service = FakeTotalReturnService()
    api_main.app.dependency_overrides[get_total_return_service] = lambda: service
    yield service
    api_main.app.dependency_overrides.clear()


@pytest.fixture(name="client")
def _client(fake_service: FakeTotalReturnService) -> TestClient:
    _ = fake_service
    return TestClient(api_main.app)


def test_upload_returns_201_with_import_summary(client: TestClient) -> None:
    response = client.post(
        "/total-returns/upload",
        files={"file": ("Total Returns.xlsx", b"content")},
    )

    assert response.status_code == 201
    assert response.json()["horizons_months"] == [3]


def test_upload_returns_422_when_nothing_parses(client: TestClient) -> None:
    response = client.post(
        "/total-returns/upload", files={"file": ("Total Returns.xlsx", b"")}
    )

    assert response.status_code == 422
    assert "No valid total-return rows" in response.json()["detail"]


def test_list_windows(client: TestClient) -> None:
    response = client.get("/total-returns/windows")

    assert response.status_code == 200
    assert response.json()["windows"][0]["horizon_months"] == 3


def test_delete_windows_forwards_the_horizon_filter(
    client: TestClient, fake_service: FakeTotalReturnService
) -> None:
    response = client.delete("/total-returns/windows", params={"horizon_months": 3})

    assert response.status_code == 200
    assert fake_service.deleted_horizons == [3]


def test_delete_windows_without_filter_deletes_all(
    client: TestClient, fake_service: FakeTotalReturnService
) -> None:
    response = client.delete("/total-returns/windows")

    assert response.status_code == 200
    assert fake_service.deleted_horizons == [None]


def test_delete_windows_returns_422_for_a_non_positive_horizon(client: TestClient) -> None:
    response = client.delete("/total-returns/windows", params={"horizon_months": 0})

    assert response.status_code == 422
