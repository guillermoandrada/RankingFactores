"""Tests for the Bloomberg individual-variable ingestion service."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from api.services.db_metric_service import DbMetricService
from modules.infrastructure.db import FinancialDatabase


def _workbook(rows: list[list], sheet_name: str = "Volatility 12m") -> bytes:
    buffer = io.BytesIO()
    pd.DataFrame(rows).to_excel(buffer, sheet_name=sheet_name, header=False, index=False)
    return buffer.getvalue()


def _variable_workbook() -> bytes:
    """Two periods: one already loaded, one new; plus a security not in the database."""
    return _workbook(
        [
            ["SPX Index", pd.Timestamp("2024-12-31"), None, None, None,
             "SPX Index", pd.Timestamp("2025-03-31"), None, None, None],
            ["#NAME?", None, None, None, None,
             "#NAME?", None, None, None, None],
            ["A UN Equity", "Agilent Technologies", "Health Care", "Health Care Equipment", 0.30,
             "A UN Equity", "Agilent Technologies", "Health Care", "Health Care Equipment", 0.31],
            ["AAPL UW Equity", "Apple Inc", "Information Technology", "Hardware", 0.20,
             "AAPL UW Equity", "Apple Inc", "Information Technology", "Hardware", 0.21],
            ["NEW UN Equity", "Newcomer Inc", "Industrials", "Machinery", 0.90,
             "NEW UN Equity", "Newcomer Inc", "Industrials", "Machinery", 0.91],
        ]
    )


@pytest.fixture(name="db")
def _db(tmp_path) -> FinancialDatabase:
    database = FinancialDatabase(db_url=f"sqlite:///{tmp_path}/metrics.db")
    database.save_fundamentals(
        pd.DataFrame(
            {
                "Ticker": ["A", "AAPL"],
                "Long Name": ["Agilent Technologies", "Apple Inc"],
                "GICS Sector Name": ["Health Care", "Information Technology"],
                "GICS Industry Group Name": ["Health Care Equipment", "Technology Hardware"],
                "Market Cap (USD)": [40e9, 3_000e9],
                "Current ROE": [0.25, 1.50],
            }
        ),
        period="2024/12/31",
        index_code="SPX Index",
    )
    return database


@pytest.fixture(name="service")
def _service(db: FinancialDatabase) -> DbMetricService:
    return DbMetricService(db=db)


def test_ingest_stores_the_variable_in_every_period_of_the_file(
    service: DbMetricService,
    db: FinancialDatabase,
) -> None:
    result = service.ingest_variable_file(_variable_workbook(), "volatility.xlsx")

    assert result["variable"] == "Volatility 12m"
    assert result["records_count"] == 6
    assert result["securities_count"] == 3
    assert [period["period"] for period in result["periods"]] == ["2024/12/31", "2025/03/31"]
    assert sorted(db.list_periods()) == ["2024/12/31", "2025/03/31"]

    content = db.get_period_content("2025/03/31")
    values = {row["ticker"]: row["Volatility 12m"] for row in content["data"]}
    assert values == {"A": 0.31, "AAPL": 0.21, "NEW": 0.91}


def test_ingest_creates_securities_that_are_not_in_the_database_yet(
    service: DbMetricService,
    db: FinancialDatabase,
) -> None:
    service.ingest_variable_file(_variable_workbook(), "volatility.xlsx")

    newcomer = next(
        row
        for row in db.get_security_metadata("2025/03/31")
        if row["ticker"] == "NEW"
    )
    assert newcomer["name"] == "Newcomer Inc"
    assert newcomer["sector"] == "Industrials"
    assert newcomer["industry"] == "Machinery"


def test_ingest_classifies_new_securities_in_a_period_that_did_not_exist(
    service: DbMetricService,
    db: FinancialDatabase,
) -> None:
    service.ingest_variable_file(_variable_workbook(), "volatility.xlsx")

    content = db.get_period_content("2025/03/31")
    sectors = {row["ticker"]: row["sector"] for row in content["data"]}
    assert sectors == {
        "A": "Health Care",
        "AAPL": "Information Technology",
        "NEW": "Industrials",
    }


def test_ingest_reports_the_index_name_of_each_period(service: DbMetricService) -> None:
    result = service.ingest_variable_file(_variable_workbook(), "volatility.xlsx")

    assert {period["index_code"] for period in result["periods"]} == {"SPX Index"}


def test_ingest_does_not_touch_index_membership(
    service: DbMetricService,
    db: FinancialDatabase,
) -> None:
    """A variable file may cover part of a universe, so membership stays as imported."""
    service.ingest_variable_file(_variable_workbook(), "volatility.xlsx")

    with db.engine.connect() as conn:
        rows = conn.exec_driver_sql(
            "select period, count(*) from index_membership group by period"
        ).fetchall()
    assert rows == [("2024/12/31", 2)]


def test_ingest_leaves_other_metrics_of_the_period_untouched(
    service: DbMetricService,
    db: FinancialDatabase,
) -> None:
    service.ingest_variable_file(_variable_workbook(), "volatility.xlsx")

    content = db.get_period_content("2024/12/31")
    assert sorted(content["metrics"]) == ["Current ROE", "Volatility 12m"]
    values = {row["ticker"]: row["Current ROE"] for row in content["data"]}
    assert values["A"] == 0.25
    assert values["AAPL"] == 1.50


def test_ingest_keeps_the_market_cap_already_stored(
    service: DbMetricService,
    db: FinancialDatabase,
) -> None:
    """The file has no market cap column; it must not blank the stored one."""
    service.ingest_variable_file(_variable_workbook(), "volatility.xlsx")

    caps = {
        row["ticker"]: row["market_cap_usd"]
        for row in db.get_security_metadata("2024/12/31")
    }
    assert caps["A"] == 40e9
    assert caps["NEW"] is None


def test_ingest_replaces_previous_values_of_the_same_variable(
    service: DbMetricService,
    db: FinancialDatabase,
) -> None:
    service.ingest_variable_file(_variable_workbook(), "volatility.xlsx")
    service.ingest_variable_file(
        _workbook(
            [
                ["SPX Index", pd.Timestamp("2024-12-31"), None, None, None],
                ["#NAME?", None, None, None, None],
                ["A UN Equity", "Agilent Technologies", "Health Care", "Equipment", 0.55],
            ]
        ),
        "volatility.xlsx",
    )

    content = db.get_period_content("2024/12/31")
    values = {row["ticker"]: row.get("Volatility 12m") for row in content["data"]}
    assert values["A"] == 0.55
    assert values["AAPL"] == 0.20


def test_ingest_rejects_a_file_without_period_blocks(service: DbMetricService) -> None:
    content = _workbook(
        [
            ["SPX Index", "x", None, None, None],
            ["#NAME?", None, None, None, None],
            ["A UN Equity", "Agilent", "Health Care", "Equipment", 0.3],
        ]
    )

    with pytest.raises(ValueError, match="No valid variable values"):
        service.ingest_variable_file(content, "volatility.xlsx")


def _values_only_workbook() -> bytes:
    """Ticker and value only: 'A' and 'AAPL' are in the database, 'NEW' is not."""
    return _workbook(
        [
            ["SPX Index", pd.Timestamp("2024-12-31"), "SPX Index", pd.Timestamp("2025-03-31")],
            ["ID", "volatility(calendar)", "ID", "volatility(calendar)"],
            ["A UN Equity", 0.30, "A UN Equity", 0.31],
            ["AAPL UW Equity", 0.20, "AAPL UW Equity", 0.21],
            ["NEW UN Equity", 0.90, "NEW UN Equity", 0.91],
        ]
    )


def test_values_only_file_never_creates_a_security(
    service: DbMetricService,
    db: FinancialDatabase,
) -> None:
    before = db.get_existing_tickers(["A", "AAPL", "NEW"])

    result = service.ingest_variable_file(_values_only_workbook(), "volatility.xlsx")

    assert result["creates_securities"] is False
    assert db.get_existing_tickers(["A", "AAPL", "NEW"]) == before == {"A", "AAPL"}


def test_values_only_file_reports_the_identifiers_it_could_not_place(
    service: DbMetricService,
) -> None:
    result = service.ingest_variable_file(_values_only_workbook(), "volatility.xlsx")

    assert result["securities_skipped"] == ["NEW"]
    assert result["securities_count"] == 2
    assert result["records_count"] == 4


def test_values_only_file_still_appends_the_metric_to_known_securities(
    service: DbMetricService,
    db: FinancialDatabase,
) -> None:
    service.ingest_variable_file(_values_only_workbook(), "volatility.xlsx")

    content = db.get_period_content("2024/12/31")
    values = {row["ticker"]: row["Volatility 12m"] for row in content["data"]}
    assert values == {"A": 0.30, "AAPL": 0.20}
    assert sorted(content["metrics"]) == ["Current ROE", "Volatility 12m"]


def test_values_only_file_leaves_names_and_classification_alone(
    service: DbMetricService,
    db: FinancialDatabase,
) -> None:
    """The file has no name or GICS columns, so nothing may be blanked."""
    before = {row["ticker"]: row for row in db.get_security_metadata("2024/12/31")}

    service.ingest_variable_file(_values_only_workbook(), "volatility.xlsx")

    after = {row["ticker"]: row for row in db.get_security_metadata("2024/12/31")}
    for ticker in ("A", "AAPL"):
        assert after[ticker]["name"] == before[ticker]["name"]
        assert after[ticker]["sector"] == before[ticker]["sector"]
        assert after[ticker]["industry"] == before[ticker]["industry"]
        assert after[ticker]["market_cap_usd"] == before[ticker]["market_cap_usd"]


def test_values_only_file_is_rejected_when_no_identifier_is_known(
    service: DbMetricService,
) -> None:
    content = _workbook(
        [
            ["SPX Index", pd.Timestamp("2024-12-31")],
            ["ID", "volatility"],
            ["NOPE UN Equity", 0.5],
        ]
    )

    with pytest.raises(ValueError, match="none of its 1 identifiers match"):
        service.ingest_variable_file(content, "volatility.xlsx")


def test_wide_file_still_creates_securities(
    service: DbMetricService,
    db: FinancialDatabase,
) -> None:
    """The narrow-layout restriction must not leak into the wide layout."""
    result = service.ingest_variable_file(_variable_workbook(), "volatility.xlsx")

    assert result["creates_securities"] is True
    assert result["securities_skipped"] == []
    assert "NEW" in db.get_existing_tickers(["NEW"])
