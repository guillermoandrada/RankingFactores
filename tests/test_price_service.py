"""Tests for the price management service."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from api.services.price_service import PriceService
from modules.infrastructure.db import FinancialDatabase
from modules.infrastructure.market_data.providers.base import PriceMatrixResult


class StubPriceProvider:
    provider_name = "stub"

    def __init__(self, closes: dict[str, float]) -> None:
        self._closes = closes

    def fetch_price_matrix(
        self,
        identifiers: list[str],
        *,
        start_date: str,
        end_date: str,
        frequency: str = "daily",
    ) -> PriceMatrixResult:
        _ = (start_date, end_date, frequency)
        columns = {i: [self._closes[i]] for i in identifiers if i in self._closes}
        missing = [i for i in identifiers if i not in columns]
        if not columns:
            return PriceMatrixResult(prices=pd.DataFrame(), missing_identifiers=missing)
        return PriceMatrixResult(
            prices=pd.DataFrame(columns, index=pd.to_datetime(["2024-05-31"])),
            resolved_identifiers={i: i for i in columns},
            missing_identifiers=missing,
        )


def _workbook(rows: list[list]) -> bytes:
    buffer = io.BytesIO()
    pd.DataFrame(rows).to_excel(buffer, header=False, index=False)
    return buffer.getvalue()


@pytest.fixture(name="service")
def _service(tmp_path) -> PriceService:
    db = FinancialDatabase(db_url=f"sqlite:///{tmp_path}/prices.db")
    return PriceService(db=db, price_provider=StubPriceProvider({"AAPL": 190.5}))


def test_ingest_reports_imported_and_skipped(service: PriceService) -> None:
    content = _workbook(
        [
            [None, None, None],
            ["Date", "AAPL US Equity", "   "],
            ["2024-01-02", 100.0, 1.0],
            ["bad date", 101.0, 1.0],
        ]
    )

    result = service.ingest_from_file(content, "prices.xlsx")

    assert result["tickers_imported"] == ["AAPL"]
    assert result["rows_written"] == 1
    assert result["rows_skipped"] == 1
    assert result["date_range"] == {"min": "2024-01-02", "max": "2024-01-02"}


def test_ingest_rejects_a_file_without_usable_rows(service: PriceService) -> None:
    content = _workbook([["header"], ["only"], ["rows"]])

    with pytest.raises(ValueError, match="No valid price rows"):
        service.ingest_from_file(content, "empty.xlsx")


def test_delete_canonicalizes_and_reports_rows(service: PriceService) -> None:
    service.ingest_from_file(
        _workbook(
            [
                [None, None],
                ["Date", "AAPL"],
                ["2024-01-02", 100.0],
                ["2024-01-03", 101.0],
            ]
        ),
        "prices.xlsx",
    )

    result = service.delete_tickers(["aapl us equity"])

    assert result == {"deleted_rows": 2, "tickers": ["AAPL"]}
    assert service.list_cached_tickers() == []


def test_delete_rejects_input_with_no_valid_tickers(service: PriceService) -> None:
    with pytest.raises(ValueError):
        service.delete_tickers(["  ", ""])


def test_latest_closes_go_through_the_injected_provider(service: PriceService) -> None:
    result = service.get_latest_closes(["AAPL", "NOPE"])

    assert result["closes"] == {"AAPL": 190.5}
    assert result["missing_identifiers"] == ["NOPE"]
