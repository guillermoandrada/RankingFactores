"""Import preview: describe a file without writing it, and report create vs replace."""

from __future__ import annotations

import io
from datetime import date

import pandas as pd
import pytest

from api.services.period_service import PeriodService
from modules.infrastructure.ingestion.importer import DataImporter


def _bloomberg_workbook(period_cell=date(2024, 3, 31)) -> bytes:
    """
    A Bloomberg-shaped sheet, matching BloombergFileReader.

    The reader takes the period from A2 and reads the table with row index 3 as the
    header, so the spacer row is load-bearing.
    """
    buffer = io.BytesIO()
    rows = [
        [None, "Universe Name", None, None],
        [period_cell, "B500", None, None],
        [None, None, None, None],
        ["Ticker", "Long Name", "GICS Sector Name", "GICS Industry Group Name"],
        ["AAA", "AAA Corp", "Tech", "Software"],
        ["BBB", "BBB Corp", "Industrials", "Machinery"],
    ]
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Data", index=False, header=False)
    return buffer.getvalue()


@pytest.fixture
def workbook_path(tmp_path):
    def _write(content: bytes, name: str = "book.xlsx") -> str:
        path = tmp_path / name
        path.write_bytes(content)
        return str(path)

    return _write


# --- DataImporter.describe_file --------------------------------------------------------


def test_describe_file_reports_shape_and_sheets(workbook_path) -> None:
    path = workbook_path(_bloomberg_workbook())

    description = DataImporter(database=object()).describe_file(path, reader="bloomberg")

    assert description["reader"] == "bloomberg"
    assert description["sheet_names"] == ["Data"]
    assert description["row_count"] == 2
    assert "Ticker" in description["columns"]
    assert description["missing_ticker_column"] is False


def test_describe_file_returns_sample_rows(workbook_path) -> None:
    path = workbook_path(_bloomberg_workbook())

    description = DataImporter(database=object()).describe_file(path, reader="bloomberg")

    assert [row["Ticker"] for row in description["sample_rows"]] == ["AAA", "BBB"]


def test_describe_file_honours_a_manual_period(workbook_path) -> None:
    path = workbook_path(_bloomberg_workbook())

    description = DataImporter(database=object()).describe_file(
        path, reader="bloomberg", period_override="2099/12/31"
    )

    assert description["period"] == "2099/12/31"
    assert description["period_source"] == "manual"


def test_describe_file_reports_undetectable_period_instead_of_raising(workbook_path) -> None:
    """For a preview, a detection failure is the answer rather than an error."""
    path = workbook_path(_bloomberg_workbook(period_cell="not a date"), "blank.xlsx")

    description = DataImporter(database=object()).describe_file(path, reader="bloomberg")

    assert description["period"] is None
    assert description["period_error"]


def test_describe_file_rejects_a_missing_path() -> None:
    with pytest.raises(FileNotFoundError):
        DataImporter(database=object()).describe_file("nope.xlsx", reader="bloomberg")


def test_describe_file_writes_nothing(workbook_path) -> None:
    """The whole point of a preview: the database is never touched."""

    class ExplodingDb:
        def __getattr__(self, name):
            raise AssertionError(f"describe_file must not call db.{name}")

    path = workbook_path(_bloomberg_workbook())
    DataImporter(database=ExplodingDb()).describe_file(path, reader="bloomberg")


# --- PeriodService.preview_period_file -------------------------------------------------


class _FakeDb:
    def __init__(self, periods: list[str]) -> None:
        self._periods = periods

    def list_periods(self) -> list[str]:
        return self._periods


def _service(periods: list[str]) -> PeriodService:
    return PeriodService(db=_FakeDb(periods), importer=DataImporter(database=object()))


def test_preview_reports_create_for_an_unknown_period() -> None:
    preview = _service([]).preview_period_file(
        _bloomberg_workbook(), "book.xlsx", reader="bloomberg"
    )

    assert preview["period"] == "2024/03/31"
    assert preview["period_exists"] is False
    assert preview["action"] == "create"


def test_preview_warns_that_an_existing_period_would_be_replaced() -> None:
    preview = _service(["2024/03/31"]).preview_period_file(
        _bloomberg_workbook(), "book.xlsx", reader="bloomberg", if_period_exists="replace"
    )

    assert preview["period_exists"] is True
    assert preview["action"] == "replace"


def test_preview_reports_append_when_merging() -> None:
    preview = _service(["2024/03/31"]).preview_period_file(
        _bloomberg_workbook(), "book.xlsx", reader="bloomberg", if_period_exists="append"
    )

    assert preview["action"] == "append"


def test_preview_applies_the_same_validation_as_create() -> None:
    service = _service([])

    with pytest.raises(ValueError, match="must be .xlsx or .xls"):
        service.preview_period_file(b"", "notes.txt", reader="bloomberg")
    with pytest.raises(ValueError, match="if_period_exists"):
        service.preview_period_file(
            _bloomberg_workbook(), "book.xlsx", reader="bloomberg", if_period_exists="merge"
        )
    with pytest.raises(ValueError, match="reader must be"):
        service.preview_period_file(_bloomberg_workbook(), "book.xlsx", reader="nonsense")
    with pytest.raises(ValueError, match="period is required"):
        service.preview_period_file(
            _bloomberg_workbook(), "book.xlsx", reader="reuters_metrics"
        )


def test_every_action_the_service_returns_has_a_ui_notice() -> None:
    """The Create tab renders an explanation per action; a new action must not fall through."""
    import pathlib

    page = pathlib.Path(__file__).resolve().parents[1] / "streamlit_app" / "pages" / "1_Periods.py"
    definitions = page.read_text(encoding="utf-8-sig").split('render_page_header("Periods"')[0]
    namespace: dict = {}
    exec(compile(definitions, str(page), "exec"), namespace)

    assert set(namespace["_ACTION_NOTICES"]) == {"create", "replace", "append"}


def test_preview_leaves_no_temporary_file_behind(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    before = set(tmp_path.iterdir())

    _service([]).preview_period_file(_bloomberg_workbook(), "book.xlsx", reader="bloomberg")

    assert set(tmp_path.iterdir()) == before
