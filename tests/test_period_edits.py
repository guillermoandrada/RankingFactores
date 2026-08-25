"""Pure helpers that turn an edited period table into API value updates."""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

_PAGE = pathlib.Path(__file__).resolve().parents[1] / "streamlit_app" / "pages" / "1_Periods.py"


@pytest.fixture(scope="module")
def page_helpers() -> dict:
    """
    Expose the page's module-level helpers without running its Streamlit body.

    The page is a script, not an importable module: everything above the first
    `render_page_header` call is definitions only, so executing that prefix is enough.
    """
    source = _PAGE.read_text(encoding="utf-8-sig")
    definitions = source.split('render_page_header("Periods"')[0]
    namespace: dict = {}
    exec(compile(definitions, str(_PAGE), "exec"), namespace)
    return namespace


@pytest.fixture
def original() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "GOOG", "AMZN"],
            "ROE": [1.0, 2.0, None, 4.0],
            "Debt": [10.0, 20.0, 30.0, 40.0],
        }
    )


def test_collects_changed_and_newly_filled_values(page_helpers, original) -> None:
    edited = original.copy()
    edited.loc[0, "ROE"] = 1.5  # changed
    edited.loc[2, "ROE"] = 3.0  # was missing, now filled

    updates, cleared, invalid = page_helpers["_collect_period_edits"](
        original, edited, ["ROE", "Debt"]
    )

    assert updates == [
        {"ticker": "AAPL", "metric_name": "ROE", "value": 1.5},
        {"ticker": "GOOG", "metric_name": "ROE", "value": 3.0},
    ]
    assert cleared == []
    assert invalid == []


def test_reports_blanked_cells_instead_of_dropping_them(page_helpers, original) -> None:
    """The API cannot store an empty value, so a blanked cell must be reported."""
    edited = original.copy()
    edited.loc[1, "ROE"] = None

    updates, cleared, invalid = page_helpers["_collect_period_edits"](
        original, edited, ["ROE", "Debt"]
    )

    assert updates == []
    assert cleared == ["MSFT · ROE"]
    assert invalid == []


def test_reports_non_numeric_cells_instead_of_dropping_them(page_helpers, original) -> None:
    edited = original.astype({"Debt": object})
    edited.loc[1, "Debt"] = "oops"

    updates, cleared, invalid = page_helpers["_collect_period_edits"](
        original, edited, ["ROE", "Debt"]
    )

    assert updates == []
    assert cleared == []
    assert invalid == ["MSFT · Debt"]


def test_untouched_table_produces_no_edits(page_helpers, original) -> None:
    assert page_helpers["_collect_period_edits"](original, original.copy(), ["ROE", "Debt"]) == (
        [],
        [],
        [],
    )


def test_missing_metric_column_is_ignored(page_helpers, original) -> None:
    updates, cleared, invalid = page_helpers["_collect_period_edits"](
        original, original.copy(), ["ROE", "Debt", "NeverUploaded"]
    )
    assert (updates, cleared, invalid) == ([], [], [])


def test_table_without_ticker_column_is_ignored(page_helpers) -> None:
    frame = pd.DataFrame({"ROE": [1.0]})
    assert page_helpers["_collect_period_edits"](frame, frame.copy(), ["ROE"]) == ([], [], [])


def test_duplicate_tickers_compare_against_the_first_row(page_helpers) -> None:
    original = pd.DataFrame({"ticker": ["AAPL", "AAPL"], "ROE": [1.0, 9.0]})
    edited = pd.DataFrame({"ticker": ["AAPL", "AAPL"], "ROE": [1.0, 1.0]})

    updates, cleared, invalid = page_helpers["_collect_period_edits"](original, edited, ["ROE"])

    assert (updates, cleared, invalid) == ([], [], [])


def test_non_numeric_original_value_does_not_raise(page_helpers) -> None:
    """A text value already in the table used to blow up the float() comparison."""
    original = pd.DataFrame({"ticker": ["AAPL"], "ROE": ["n/a"]})
    edited = pd.DataFrame({"ticker": ["AAPL"], "ROE": [2.0]})

    updates, cleared, invalid = page_helpers["_collect_period_edits"](original, edited, ["ROE"])

    assert updates == [{"ticker": "AAPL", "metric_name": "ROE", "value": 2.0}]
    assert (cleared, invalid) == ([], [])
