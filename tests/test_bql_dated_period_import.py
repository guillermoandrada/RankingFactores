from __future__ import annotations

from io import BytesIO

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
import api.routers.periods as periods_router
from api.services.period_service import PeriodService
from modules.infrastructure.db.repository import FinancialDatabase
from modules.infrastructure.ingestion.file_reader import FileReader
from modules.infrastructure.ingestion.importer import DataImporter
from modules.infrastructure.ingestion.readers.bql_dated import BqlDatedFileReader
from modules.domain.models import ImportResult

_AS_OF_DATE = pd.Timestamp("2024-03-31")
_EARLIER_DATE = pd.Timestamp("2024-02-29")


def _sheet_with_formula_row(frame: pd.DataFrame, formulas: dict[str, str]) -> pd.DataFrame:
    """Prepend the BQL formula row every plain sheet carries under its header."""
    return pd.concat([pd.DataFrame([formulas]), frame], ignore_index=True)


def _plain_sheets() -> dict[str, pd.DataFrame]:
    names = _sheet_with_formula_row(
        pd.DataFrame(
            {
                "Bloomberg Code": ["AAA US Equity", "BBB US Equity"],
                "Security Name": ["AAA Corp", "BBB Corp"],
            }
        ),
        {"Bloomberg Code": "FORMULA", "Security Name": "=NAME_FORMULA"},
    )
    classification = _sheet_with_formula_row(
        pd.DataFrame(
            {
                "Bloomberg Code": ["AAA US Equity", "BBB US Equity"],
                "Sector": ["Tech", "Industrials"],
                "Industry": ["Software", "Machinery"],
            }
        ),
        {
            "Bloomberg Code": "FORMULA",
            "Sector": "=SECTOR_FORMULA",
            "Industry": "=INDUSTRY_FORMULA",
        },
    )
    current = _sheet_with_formula_row(
        pd.DataFrame(
            {
                "Bloomberg Code": ["AAA US Equity", "BBB US Equity"],
                "Current Factor": [1.5, 2.5],
            }
        ),
        {"Bloomberg Code": "FORMULA", "Current Factor": "=CURRENT_FORMULA"},
    )
    past = _sheet_with_formula_row(
        pd.DataFrame(
            {
                "Bloomberg Code": ["AAA US Equity", "BBB US Equity"],
                "Past Factor": [10.0, 20.0],
            }
        ),
        {"Bloomberg Code": "FORMULA", "Past Factor": "=PAST_FORMULA"},
    )
    return {
        "Name": names,
        "Classification": classification,
        "Current": current,
        "Past": past,
    }


def _dated_estimated_sheet() -> pd.DataFrame:
    """
    'Estimated' as Bloomberg returns it with fill=prev: a shared DATES column and one
    column block per ticker, each value sitting on the row of the date it was reported.
    'Estimated Margin' is only reported on the earlier date, so it must import as NA.
    """
    return pd.DataFrame(
        [
            ["Ticker", "Estimated Factor", "Estimated Margin", None, None],
            ["", "AAA US Equity", "AAA US Equity", "BBB US Equity", "BBB US Equity"],
            [
                "DATES",
                "estimated_factor(fill=prev,as_of_date=2024-03-31)",
                "estimated_margin(fill=prev,as_of_date=2024-03-31)",
                "estimated_factor(fill=prev,as_of_date=2024-03-31)",
                "estimated_margin(fill=prev,as_of_date=2024-03-31)",
            ],
            [_AS_OF_DATE, 100.0, None, 200.0, None],
            [_EARLIER_DATE, None, 7.5, None, 8.5],
        ]
    )


def _plain_estimated_sheet() -> pd.DataFrame:
    return _sheet_with_formula_row(
        pd.DataFrame(
            {
                "Bloomberg Code": ["AAA US Equity", "BBB US Equity"],
                "Estimated Factor": [100.0, 200.0],
            }
        ),
        {"Bloomberg Code": "FORMULA", "Estimated Factor": "=ESTIMATED_FORMULA"},
    )


def _build_workbook_bytes(
    *,
    estimated: pd.DataFrame | None = None,
    config_date: pd.Timestamp = _AS_OF_DATE,
) -> bytes:
    """Write a BQL workbook whose 'Estimated' sheet is dated unless told otherwise."""
    estimated_sheet = _dated_estimated_sheet() if estimated is None else estimated
    dated = estimated is None
    config = pd.DataFrame([["As Of Date", config_date], ["Universe", "BQL_INDEX"]])

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, frame in _plain_sheets().items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
        estimated_sheet.to_excel(
            writer, sheet_name="Estimated", index=False, header=not dated
        )
        config.to_excel(writer, sheet_name="Config", index=False, header=False)
    return buffer.getvalue()


def _seed_security_metadata(db: FinancialDatabase) -> None:
    seed_df = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB"],
            "Long Name": ["AAA Corp", "BBB Corp"],
            "GICS Sector Name": ["Tech", "Industrials"],
            "GICS Industry Group Name": ["Software", "Machinery"],
            "Market Cap (USD)": [1_000.0, 2_000.0],
            "Seed Metric": [5.0, 6.0],
        }
    )
    db.save_fundamentals(seed_df, "2023/12/31", index_code="SEED", mode="replace")


def test_bql_dated_reader_keeps_only_values_reported_on_the_config_date(tmp_path) -> None:
    """A value carried over from an earlier date is the bug this reader exists for."""
    file_path = tmp_path / "bql_dated.xlsx"
    file_path.write_bytes(_build_workbook_bytes())

    reader = BqlDatedFileReader()
    result = reader.read(str(file_path))

    assert list(result.columns) == [
        "Ticker",
        "Long Name",
        "GICS Sector Name",
        "GICS Industry Group Name",
        "Current Factor",
        "Past Factor",
        "Estimated Factor",
        "Estimated Margin",
    ]
    assert result["Ticker"].tolist() == ["AAA", "BBB"]
    assert result["Current Factor"].tolist() == [1.5, 2.5]
    assert result["Estimated Factor"].tolist() == [100.0, 200.0]
    assert result["Estimated Margin"].isna().all()
    assert reader.extract_period(str(file_path)) == "2024/03/31"
    assert reader.extract_index_code(str(file_path)) == "BQL_INDEX"


def test_bql_dated_reader_is_the_one_auto_detection_picks(tmp_path) -> None:
    """'bql' also accepts a dated workbook, so detection order is what protects the read."""
    file_path = tmp_path / "bql_dated_auto.xlsx"
    file_path.write_bytes(_build_workbook_bytes())

    result = FileReader().read(str(file_path), "auto")

    assert result["Estimated Factor"].tolist() == [100.0, 200.0]
    assert result["Estimated Margin"].isna().all()


def test_bql_dated_reader_rejects_a_plain_estimated_sheet(tmp_path) -> None:
    file_path = tmp_path / "bql_plain.xlsx"
    file_path.write_bytes(_build_workbook_bytes(estimated=_plain_estimated_sheet()))

    reader = BqlDatedFileReader()

    assert reader.can_read(str(file_path)) is False
    with pytest.raises(ValueError, match="must be in dated layout"):
        reader.read(str(file_path))


def test_bql_dated_reader_fails_when_no_row_carries_the_config_date(tmp_path) -> None:
    file_path = tmp_path / "bql_dated_off_period.xlsx"
    file_path.write_bytes(_build_workbook_bytes(config_date=pd.Timestamp("2024-06-30")))

    with pytest.raises(ValueError, match="no row reported on 2024-06-30"):
        BqlDatedFileReader().read(str(file_path))


def test_bql_dated_import_uses_config_index_code(tmp_path) -> None:
    db_path = tmp_path / "bql_dated.db"
    db = FinancialDatabase(db_url=f"sqlite:///{db_path}")
    _seed_security_metadata(db)

    file_path = tmp_path / "bql_dated.xlsx"
    file_path.write_bytes(_build_workbook_bytes())

    importer = DataImporter(database=db)
    result = importer.import_file(
        str(file_path),
        verbose=False,
        reader="bql_dated",
    )

    assert result.period == "2024/03/31"
    assert result.index_code == "BQL_INDEX"

    content = db.get_period_content("2024/03/31")
    assert content["data"][0]["ticker"] == "AAA"
    assert content["data"][0]["Estimated Factor"] == 100.0
    assert content["data"][0].get("Estimated Margin") is None


def test_period_service_accepts_the_bql_dated_reader() -> None:
    class FakeImporter:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def import_file(self, filepath: str, **kwargs):
            self.calls.append({"filepath": filepath, **kwargs})
            return ImportResult(
                period="2024/03/31",
                companies_count=2,
                metrics_count=4,
                records_count=8,
                index_code="BQL_INDEX",
            )

    class FakeDb:
        def list_periods(self) -> list[str]:
            return ["2023/12/31"]

    importer = FakeImporter()
    service = PeriodService(db=FakeDb(), importer=importer)

    result = service.create_period_from_file(
        file_contents=_build_workbook_bytes(),
        filename="bql_dated.xlsx",
        if_period_exists="replace",
        reader="bql_dated",
    )

    assert result["success"] is True
    assert result["action"] == "create"
    assert importer.calls[0]["reader"] == "bql_dated"


def test_periods_endpoint_accepts_the_bql_dated_reader(monkeypatch) -> None:
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

    monkeypatch.setattr(periods_router, "get_period_service", lambda: FakeService())
    client = TestClient(api_main.app)

    response = client.post(
        "/periods?reader=bql_dated&if_period_exists=append",
        files={
            "file": (
                "bql_dated.xlsx",
                _build_workbook_bytes(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )

    assert response.status_code == 201
    assert response.json()["reader"] == "bql_dated"
