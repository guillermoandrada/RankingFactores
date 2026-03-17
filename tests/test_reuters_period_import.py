from __future__ import annotations

from io import BytesIO

import pandas as pd
from fastapi.testclient import TestClient

import api.main as api_main
import api.routers.periods as periods_router
from modules.db.repository import FinancialDatabase
from api.services.period_service import PeriodService
from modules.ingestion.readers.reuters_metrics import ReutersMetricsFileReader
from modules.models import ImportResult


def _build_reuters_excel_bytes() -> bytes:
    df = pd.DataFrame(
        {
            "Identifier (RIC)": ["VIK.N", "GEV.N"],
            "Company Name": ["Viking Holdings Ltd", "GE Vernova Inc"],
            "GICS Sector Name": ["Financials", "Information Technology"],
            "GICS Industry Group Name": ["Financial Services", "Semiconductors & Semiconductor Equipment"],
            "Company Market Cap (Millions, USD)": [46_384.49, 117_901.42],
            "Earnings Quality Country Rank, Current": [60, 92],
        }
    )
    buffer = BytesIO()
    df.to_excel(buffer, index=False)
    return buffer.getvalue()


def test_reuters_reader_normalizes_columns(tmp_path) -> None:
    file_path = tmp_path / "reuters.xlsx"
    file_path.write_bytes(_build_reuters_excel_bytes())

    reader = ReutersMetricsFileReader()

    result = reader.read(str(file_path))

    assert list(result.columns) == [
        "Ticker",
        "Long Name",
        "GICS Sector Name",
        "GICS Industry Group Name",
        "Market Cap (USD)",
        "Reuters Score",
    ]
    assert result["Ticker"].tolist() == ["VIK", "GEV"]
    assert result["Long Name"].tolist() == ["Viking Holdings Ltd", "GE Vernova Inc"]
    assert result["Reuters Score"].tolist() == [60, 92]
    assert result["Market Cap (USD)"].tolist() == [46_384_490_000.0, 117_901_420_000.0]


def test_period_service_passes_reader_and_manual_period() -> None:
    class FakeImporter:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def import_file(self, filepath: str, **kwargs):
            self.calls.append({"filepath": filepath, **kwargs})
            return ImportResult(
                period="2024 Q1",
                companies_count=2,
                metrics_count=1,
                records_count=2,
                index_code=None,
            )

    importer = FakeImporter()
    service = PeriodService(db=object(), importer=importer)

    result = service.create_period_from_file(
        file_contents=_build_reuters_excel_bytes(),
        filename="reuters.xlsx",
        if_period_exists="append",
        reader="reuters_metrics",
        period="2024 Q1",
    )

    assert result["success"] is True
    assert importer.calls
    assert importer.calls[0]["reader"] == "reuters_metrics"
    assert importer.calls[0]["period_override"] == "2024 Q1"
    assert importer.calls[0]["if_period_exists"] == "append"


def test_periods_endpoint_requires_period_for_reuters() -> None:
    client = TestClient(api_main.app)

    response = client.post(
        "/periods?reader=reuters_metrics&if_period_exists=replace",
        files={
            "file": (
                "reuters.xlsx",
                _build_reuters_excel_bytes(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )

    assert response.status_code == 400
    assert "period is required" in response.json()["detail"]


def test_periods_endpoint_reuters_replace(monkeypatch) -> None:
    class FakeService:
        def create_period_from_file(
            self,
            *,
            file_contents: bytes,
            filename: str,
            if_period_exists: str,
            reader: str,
            period: str | None = None,
        ):
            return {
                "success": True,
                "period": period,
                "reader": reader,
                "if_period_exists": if_period_exists,
                "bytes": len(file_contents),
                "filename": filename,
            }

    def _fake_service():
        return FakeService()

    monkeypatch.setattr(periods_router, "get_period_service", _fake_service)
    client = TestClient(api_main.app)

    response = client.post(
        "/periods?reader=reuters_metrics&period=2024%20Q1&if_period_exists=replace",
        files={
            "file": (
                "reuters.xlsx",
                _build_reuters_excel_bytes(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["period"] == "2024 Q1"
    assert payload["reader"] == "reuters_metrics"
    assert payload["if_period_exists"] == "replace"


def test_periods_endpoint_reuters_append(monkeypatch) -> None:
    class FakeService:
        def create_period_from_file(
            self,
            *,
            file_contents: bytes,
            filename: str,
            if_period_exists: str,
            reader: str,
            period: str | None = None,
        ):
            return {
                "success": True,
                "period": period,
                "reader": reader,
                "if_period_exists": if_period_exists,
                "bytes": len(file_contents),
                "filename": filename,
            }

    def _fake_service():
        return FakeService()

    monkeypatch.setattr(periods_router, "get_period_service", _fake_service)
    client = TestClient(api_main.app)

    response = client.post(
        "/periods?reader=reuters_metrics&period=2024%20Q1&if_period_exists=append",
        files={
            "file": (
                "reuters.xlsx",
                _build_reuters_excel_bytes(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["period"] == "2024 Q1"
    assert payload["reader"] == "reuters_metrics"
    assert payload["if_period_exists"] == "append"


def test_reuters_append_preserves_existing_period_classification(tmp_path) -> None:
    db_path = tmp_path / "reuters_append.db"
    db = FinancialDatabase(db_url=f"sqlite:///{db_path}")

    base_period = pd.DataFrame(
        {
            "Ticker": ["AAA"],
            "Long Name": ["AAA Corp"],
            "GICS Sector Name": ["Tech"],
            "GICS Industry Group Name": ["Software"],
            "Market Cap (USD)": [1000.0],
            "Metric A": [1.0],
        }
    )
    db.save_fundamentals(base_period, "2024 Q1", mode="replace")

    reuters_append = pd.DataFrame(
        {
            "Ticker": ["AAA"],
            "Long Name": ["AAA Corp"],
            "GICS Sector Name": ["Financials"],
            "GICS Industry Group Name": ["Banks"],
            "Market Cap (USD)": [1000.0],
            "Reuters Score": [88.0],
        }
    )
    db.save_fundamentals(
        reuters_append,
        "2024 Q1",
        mode="append",
        preserve_existing_classification=True,
    )

    content = db.get_period_content("2024 Q1")

    assert content["metrics"] == ["Metric A", "Reuters Score"]
    assert content["data"][0]["ticker"] == "AAA"
    assert content["data"][0]["sector"] == "Tech"
    assert content["data"][0]["industry"] == "Software"
    assert content["data"][0]["Reuters Score"] == 88.0
