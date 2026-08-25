"""Tests for the DB metrics router."""

from __future__ import annotations

import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.dependencies import get_db_metric_service


class FakeDbMetricService:
    def __init__(self) -> None:
        self.requested_sheets: list[str | None] = []

    def ingest_variable_file(
        self,
        file_content: bytes,
        filename: str,
        *,
        sheet_name: str | None = None,
    ) -> dict:
        self.requested_sheets.append(sheet_name)
        if not file_content:
            raise ValueError(f"No valid variable values found in '{filename}'.")
        return {
            "variable": "Volatility 12m",
            "periods": [
                {
                    "period": "2024/12/31",
                    "index_code": "SPX Index",
                    "companies_count": 2,
                    "metrics_count": 1,
                    "records_count": 2,
                },
            ],
            "securities_count": 2,
            "records_count": 2,
            "periods_skipped": [],
            "rows_skipped": 0,
        }


@pytest.fixture(name="fake_service")
def _fake_service() -> FakeDbMetricService:
    service = FakeDbMetricService()
    api_main.app.dependency_overrides[get_db_metric_service] = lambda: service
    yield service
    api_main.app.dependency_overrides.clear()


@pytest.fixture(name="client")
def _client(fake_service: FakeDbMetricService) -> TestClient:
    _ = fake_service
    return TestClient(api_main.app)


def _xlsx_bytes() -> bytes:
    buffer = io.BytesIO()
    pd.DataFrame(
        [
            ["SPX Index", "2024-12-31", None, None, None],
            ["#NAME?", None, None, None, None],
            ["A UN Equity", "Agilent", "Health Care", "Equipment", 0.3],
        ]
    ).to_excel(buffer, header=False, index=False)
    return buffer.getvalue()


def test_upload_returns_201_with_import_summary(client: TestClient) -> None:
    response = client.post(
        "/db-metrics",
        files={"file": ("volatility.xlsx", _xlsx_bytes())},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["variable"] == "Volatility 12m"
    assert payload["records_count"] == 2
    assert payload["periods"][0]["period"] == "2024/12/31"


def test_upload_forwards_the_requested_sheet(
    client: TestClient,
    fake_service: FakeDbMetricService,
) -> None:
    response = client.post(
        "/db-metrics",
        params={"sheet": "Beta 12m"},
        files={"file": ("variable.xlsx", _xlsx_bytes())},
    )

    assert response.status_code == 201
    assert fake_service.requested_sheets == ["Beta 12m"]


def test_upload_rejects_a_non_excel_file(client: TestClient) -> None:
    response = client.post(
        "/db-metrics",
        files={"file": ("variable.csv", b"ticker,value")},
    )

    assert response.status_code == 400


def test_upload_returns_422_when_nothing_parses(client: TestClient) -> None:
    response = client.post(
        "/db-metrics",
        files={"file": ("variable.xlsx", b"")},
    )

    assert response.status_code == 422
    assert "No valid variable values" in response.json()["detail"]
