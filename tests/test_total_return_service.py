"""Tests for the total-return management service."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from api.services.total_return_service import TotalReturnService
from modules.infrastructure.db import FinancialDatabase
from tests.test_bloomberg_total_returns_reader import total_return_block, workbook


@pytest.fixture(name="db")
def _db(tmp_path) -> FinancialDatabase:
    return FinancialDatabase(db_url=f"sqlite:///{tmp_path}/total_returns.db")


@pytest.fixture(name="service")
def _service(db: FinancialDatabase) -> TotalReturnService:
    return TotalReturnService(db=db)


def _two_horizon_file(aapl_return: float = 0.1) -> bytes:
    return workbook(
        {
            "3m": total_return_block(
                dt.date(2025, 6, 30),
                dt.date(2025, 3, 30),
                [("AAPL UW Equity", aapl_return), ("MSFT UW Equity", 0.2)],
            ),
            "12m": total_return_block(
                dt.date(2025, 6, 30), dt.date(2024, 6, 30), [("AAPL UW Equity", 0.5)]
            ),
        }
    )


def test_ingest_reports_the_import_summary(service: TotalReturnService) -> None:
    result = service.ingest_from_file(_two_horizon_file(), "Total Returns.xlsx")

    assert result["tickers_imported"] == ["AAPL", "MSFT"]
    assert result["rows_written"] == 3
    assert result["horizons_months"] == [3, 12]
    assert result["windows_imported"] == 2
    assert result["date_range"] == {"min": "2024-06-30", "max": "2025-06-30"}


def test_ingest_rejects_a_file_without_blocks(service: TotalReturnService) -> None:
    content = workbook({"prices": [[None, None], ["Date", "AAPL"], ["2024-01-02", 1.0]]})

    with pytest.raises(ValueError, match="No valid total-return rows"):
        service.ingest_from_file(content, "prices.xlsx")


def test_reupload_replaces_the_window_instead_of_duplicating_it(
    service: TotalReturnService, db: FinancialDatabase
) -> None:
    service.ingest_from_file(_two_horizon_file(0.1), "v1.xlsx")
    service.ingest_from_file(_two_horizon_file(0.9), "v2.xlsx")

    stored = pd.read_sql_table("total_returns", db.engine)
    assert len(stored) == 3
    is_aapl_3m = (stored["ticker"] == "AAPL") & (stored["horizon_months"] == 3)
    assert stored.loc[is_aapl_3m, "total_return"].tolist() == [0.9]


def test_list_windows_groups_by_horizon_and_window(service: TotalReturnService) -> None:
    service.ingest_from_file(_two_horizon_file(), "Total Returns.xlsx")

    assert service.list_windows() == [
        {
            "horizon_months": 3,
            "start_date": "2025-03-30",
            "end_date": "2025-06-30",
            "ticker_count": 2,
        },
        {
            "horizon_months": 12,
            "start_date": "2024-06-30",
            "end_date": "2025-06-30",
            "ticker_count": 1,
        },
    ]


def test_delete_windows_can_target_one_horizon(service: TotalReturnService) -> None:
    service.ingest_from_file(_two_horizon_file(), "Total Returns.xlsx")

    result = service.delete_windows(horizon_months=3)

    assert result == {"deleted_rows": 2, "horizon_months": 3}
    assert [window["horizon_months"] for window in service.list_windows()] == [12]


def test_delete_windows_rejects_a_non_positive_horizon(service: TotalReturnService) -> None:
    with pytest.raises(ValueError):
        service.delete_windows(horizon_months=0)
