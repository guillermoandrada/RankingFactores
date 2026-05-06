from __future__ import annotations

from io import BytesIO

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
import api.routers.periods as periods_router
from api.services.period_service import PeriodService
from modules.db.repository import FinancialDatabase
from modules.ingestion.importer import DataImporter
from modules.ingestion.readers.bql import BqlFileReader
from modules.models import ImportResult


def _build_bql_excel_bytes() -> bytes:
    names = pd.DataFrame(
        {
            "Bloomberg Code": ["AAA US Equity", "BBB US Equity"],
            "Name": ["AAA Corp", "BBB Corp"],
        }
    )
    names = pd.concat(
        [
            pd.DataFrame(
                {
                    "Bloomberg Code": ["FORMULA"],
                    "Name": ["=NAME_FORMULA"],
                }
            ),
            names,
        ],
        ignore_index=True,
    )
    classification = pd.DataFrame(
        {
            "Bloomberg Code": ["AAA US Equity", "BBB US Equity"],
            "Sector": ["Tech", "Industrials"],
            "Industry": ["Software", "Machinery"],
        }
    )
    classification = pd.concat(
        [
            pd.DataFrame(
                {
                    "Bloomberg Code": ["FORMULA"],
                    "Sector": ["=SECTOR_FORMULA"],
                    "Industry": ["=INDUSTRY_FORMULA"],
                }
            ),
            classification,
        ],
        ignore_index=True,
    )
    current = pd.DataFrame(
        {
            "Bloomberg Code": ["AAA US Equity", "BBB US Equity"],
            "Current Factor": [1.5, 2.5],
        }
    )
    current = pd.concat(
        [
            pd.DataFrame(
                {
                    "Bloomberg Code": ["FORMULA"],
                    "Current Factor": ["=CURRENT_FORMULA"],
                }
            ),
            current,
        ],
        ignore_index=True,
    )
    past = pd.DataFrame(
        {
            "Bloomberg Code": ["AAA US Equity", "BBB US Equity"],
            "Past Factor": [10.0, 20.0],
        }
    )
    past = pd.concat(
        [
            pd.DataFrame(
                {
                    "Bloomberg Code": ["FORMULA"],
                    "Past Factor": ["=PAST_FORMULA"],
                }
            ),
            past,
        ],
        ignore_index=True,
    )
    estimated = pd.DataFrame(
        {
            "Bloomberg Code": ["AAA US Equity", "BBB US Equity"],
            "Estimated Factor": [100.0, 200.0],
        }
    )
    estimated = pd.concat(
        [
            pd.DataFrame(
                {
                    "Bloomberg Code": ["FORMULA"],
                    "Estimated Factor": ["=ESTIMATED_FORMULA"],
                }
            ),
            estimated,
        ],
        ignore_index=True,
    )
    config = pd.DataFrame(
        [
            ["As Of Date", pd.Timestamp("2024-03-31")],
            ["Universe", "BQL_INDEX"],
        ]
    )

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        names.to_excel(writer, sheet_name="Name", index=False)
        classification.to_excel(writer, sheet_name="Classification", index=False)
        current.to_excel(writer, sheet_name="Current", index=False)
        past.to_excel(writer, sheet_name="Past", index=False)
        estimated.to_excel(writer, sheet_name="Estimated", index=False)
        config.to_excel(writer, sheet_name="Config", index=False, header=False)
    return buffer.getvalue()


def _seed_security_metadata(db: FinancialDatabase, *, market_caps: list[float | None]) -> None:
    seed_df = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB"],
            "Long Name": ["AAA Corp", "BBB Corp"],
            "GICS Sector Name": ["Tech", "Industrials"],
            "GICS Industry Group Name": ["Software", "Machinery"],
            "Market Cap (USD)": market_caps,
            "Seed Metric": [5.0, 6.0],
        }
    )
    db.save_fundamentals(seed_df, "2023/12/31", index_code="SEED", mode="replace")


def test_bql_reader_merges_three_sheets_and_extracts_period(tmp_path) -> None:
    file_path = tmp_path / "bql.xlsx"
    file_path.write_bytes(_build_bql_excel_bytes())

    reader = BqlFileReader()
    result = reader.read(str(file_path))

    assert list(result.columns) == [
        "Ticker",
        "Long Name",
        "GICS Sector Name",
        "GICS Industry Group Name",
        "Current Factor",
        "Past Factor",
        "Estimated Factor",
    ]
    assert result["Ticker"].tolist() == ["AAA", "BBB"]
    assert result["Long Name"].tolist() == ["AAA Corp", "BBB Corp"]
    assert result["GICS Sector Name"].tolist() == ["Tech", "Industrials"]
    assert result["GICS Industry Group Name"].tolist() == ["Software", "Machinery"]
    assert result["Current Factor"].tolist() == [1.5, 2.5]
    assert result["Past Factor"].tolist() == [10.0, 20.0]
    assert result["Estimated Factor"].tolist() == [100.0, 200.0]
    assert reader.extract_period(str(file_path)) == "2024/03/31"
    assert reader.extract_index_code(str(file_path)) == "BQL_INDEX"


def test_bql_import_uses_config_index_code(tmp_path) -> None:
    db_path = tmp_path / "bql.db"
    db = FinancialDatabase(db_url=f"sqlite:///{db_path}")
    _seed_security_metadata(db, market_caps=[1_000.0, 2_000.0])

    file_path = tmp_path / "bql.xlsx"
    file_path.write_bytes(_build_bql_excel_bytes())

    importer = DataImporter(database=db)
    result = importer.import_file(
        str(file_path),
        verbose=False,
        reader="bql",
    )

    assert result.period == "2024/03/31"
    assert result.index_code == "BQL_INDEX"
    assert result.metrics_count == 3

    content = db.get_period_content("2024/03/31")
    assert content["metrics"] == ["Current Factor", "Estimated Factor", "Past Factor"]
    assert content["data"][0]["ticker"] == "AAA"
    assert content["data"][0]["name"] == "AAA Corp"
    assert content["data"][0]["sector"] == "Tech"
    assert content["data"][0]["industry"] == "Software"
    assert content["data"][0]["Current Factor"] == 1.5
    assert sorted(db.list_indices()) == ["BQL_INDEX", "SEED"]


def test_bql_import_fails_when_market_cap_is_missing_for_ticker(tmp_path) -> None:
    db_path = tmp_path / "bql_missing_market_cap_for_ticker.db"
    db = FinancialDatabase(db_url=f"sqlite:///{db_path}")
    seed_df = pd.DataFrame(
        {
            "Ticker": ["AAA"],
            "Long Name": ["AAA Corp"],
            "GICS Sector Name": ["Tech"],
            "GICS Industry Group Name": ["Software"],
            "Market Cap (USD)": [1_000.0],
            "Seed Metric": [5.0],
        }
    )
    db.save_fundamentals(seed_df, "2023/12/31", mode="replace")

    file_path = tmp_path / "bql.xlsx"
    file_path.write_bytes(_build_bql_excel_bytes())

    importer = DataImporter(database=db)
    with pytest.raises(ValueError, match="Incomplete metadata"):
        importer.import_file(
            str(file_path),
            verbose=False,
            reader="bql",
        )


def test_bql_import_fails_when_market_cap_metadata_is_missing(tmp_path) -> None:
    db_path = tmp_path / "bql_missing_market_cap.db"
    db = FinancialDatabase(db_url=f"sqlite:///{db_path}")
    _seed_security_metadata(db, market_caps=[1_000.0, None])

    file_path = tmp_path / "bql.xlsx"
    file_path.write_bytes(_build_bql_excel_bytes())

    importer = DataImporter(database=db)
    with pytest.raises(ValueError, match="Incomplete metadata"):
        importer.import_file(
            str(file_path),
            verbose=False,
            reader="bql",
        )


def test_period_service_lets_bql_reader_resolve_index_code() -> None:
    class FakeImporter:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def import_file(self, filepath: str, **kwargs):
            self.calls.append({"filepath": filepath, **kwargs})
            return ImportResult(
                period="2024/03/31",
                companies_count=2,
                metrics_count=3,
                records_count=6,
                index_code="BQL_INDEX",
            )

    importer = FakeImporter()
    service = PeriodService(db=object(), importer=importer)

    result = service.create_period_from_file(
        file_contents=_build_bql_excel_bytes(),
        filename="bql.xlsx",
        if_period_exists="replace",
        reader="bql",
    )

    assert result["success"] is True
    assert importer.calls
    assert importer.calls[0]["reader"] == "bql"
    assert importer.calls[0]["index_code_override"] is None


def test_periods_endpoint_bql_no_longer_requires_manual_index_code(monkeypatch) -> None:
    class FakeService:
        def create_period_from_file(
            self,
            *,
            file_contents: bytes,
            filename: str,
            if_period_exists: str,
            reader: str,
            period: str | None = None,
            index_code: str | None = None,
        ):
            return {
                "success": True,
                "period": period,
                "reader": reader,
                "if_period_exists": if_period_exists,
                "index_code": index_code,
                "bytes": len(file_contents),
                "filename": filename,
            }

    def _fake_service():
        return FakeService()

    monkeypatch.setattr(periods_router, "get_period_service", _fake_service)
    client = TestClient(api_main.app)

    response = client.post(
        "/periods?reader=bql&if_period_exists=append",
        files={
            "file": (
                "bql.xlsx",
                _build_bql_excel_bytes(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["reader"] == "bql"
    assert payload["if_period_exists"] == "append"
    assert payload["index_code"] is None
