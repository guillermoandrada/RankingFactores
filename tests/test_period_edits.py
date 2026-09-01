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


# --- Editor column configuration -------------------------------------------------------


def test_identity_columns_are_locked(page_helpers) -> None:
    """The grid must not offer edits the save path silently ignores."""
    config = page_helpers["_period_editor_column_config"](
        ["ticker", "name", "sector", "industry", "security_id", "ROE"]
    )

    for column in ("ticker", "name", "sector", "industry"):
        assert config[column]["disabled"] is True


def test_internal_security_id_is_hidden(page_helpers) -> None:
    config = page_helpers["_period_editor_column_config"](["ticker", "security_id", "ROE"])

    assert config["security_id"] is None


def test_metric_columns_stay_editable(page_helpers) -> None:
    config = page_helpers["_period_editor_column_config"](["ticker", "ROE", "Debt"])

    assert "ROE" not in config
    assert "Debt" not in config


def test_ticker_is_pinned_but_other_identity_columns_are_not(page_helpers) -> None:
    """Ticker stays in view while scrolling a 500-row period."""
    config = page_helpers["_period_editor_column_config"](["ticker", "name", "ROE"])

    assert config["ticker"]["pinned"] is True
    assert config["name"]["pinned"] is False


def test_absent_columns_are_not_configured(page_helpers) -> None:
    config = page_helpers["_period_editor_column_config"](["ticker", "ROE"])

    assert set(config) == {"ticker"}


# --- Row filtering ---------------------------------------------------------------------


@pytest.fixture
def universe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "XOM", "BRK.B"],
            "name": ["Apple Inc", "Microsoft Corp", "Exxon Mobil", "Berkshire Hathaway"],
            "sector": ["Tech", "Tech", "Energy", "Financials"],
            "ROE": [1.0, None, 3.0, 4.0],
            "Debt": [10.0, 20.0, None, 40.0],
        }
    )


def _tickers(df: pd.DataFrame) -> list[str]:
    return df["ticker"].tolist()


def test_search_matches_ticker_or_name(page_helpers, universe) -> None:
    filter_rows = page_helpers["_filter_period_rows"]
    common = {"sector": "", "only_missing": False, "metric_names": ["ROE", "Debt"]}

    assert _tickers(filter_rows(universe, search="aapl", **common)) == ["AAPL"]
    assert _tickers(filter_rows(universe, search="Exxon", **common)) == ["XOM"]


def test_search_is_case_insensitive_and_partial(page_helpers, universe) -> None:
    result = page_helpers["_filter_period_rows"](
        universe, search="corp", sector="", only_missing=False, metric_names=["ROE"]
    )
    assert _tickers(result) == ["MSFT"]


def test_search_treats_input_as_literal_text(page_helpers, universe) -> None:
    """A ticker like BRK.B, or a stray bracket, must not be read as a regex."""
    filter_rows = page_helpers["_filter_period_rows"]
    common = {"sector": "", "only_missing": False, "metric_names": ["ROE"]}

    assert _tickers(filter_rows(universe, search="BRK.B", **common)) == ["BRK.B"]
    # "BRK?B" would match BRK.B if the pattern were a regex.
    assert _tickers(filter_rows(universe, search="BRK?B", **common)) == []
    # An unbalanced bracket is a regex compile error, not a match.
    assert _tickers(filter_rows(universe, search="(", **common)) == []


def test_blank_search_keeps_every_row(page_helpers, universe) -> None:
    result = page_helpers["_filter_period_rows"](
        universe, search="   ", sector="", only_missing=False, metric_names=["ROE"]
    )
    assert len(result) == len(universe)


def test_sector_filter(page_helpers, universe) -> None:
    result = page_helpers["_filter_period_rows"](
        universe, search="", sector="Tech", only_missing=False, metric_names=["ROE"]
    )
    assert _tickers(result) == ["AAPL", "MSFT"]


def test_only_missing_keeps_rows_with_any_gap(page_helpers, universe) -> None:
    result = page_helpers["_filter_period_rows"](
        universe, search="", sector="", only_missing=True, metric_names=["ROE", "Debt"]
    )
    assert _tickers(result) == ["MSFT", "XOM"]


def test_only_missing_respects_the_metric_subset(page_helpers, universe) -> None:
    result = page_helpers["_filter_period_rows"](
        universe, search="", sector="", only_missing=True, metric_names=["ROE"]
    )
    assert _tickers(result) == ["MSFT"]


def test_filters_combine(page_helpers, universe) -> None:
    result = page_helpers["_filter_period_rows"](
        universe, search="", sector="Tech", only_missing=True, metric_names=["ROE", "Debt"]
    )
    assert _tickers(result) == ["MSFT"]


# --- Editor identity -------------------------------------------------------------------


def test_editor_key_changes_with_every_filter(page_helpers) -> None:
    """A stale key would replay a pending edit onto whichever row took that position."""
    suffix = page_helpers["_editor_key_suffix"]
    baseline = suffix("2024Q1", "", "", False)

    assert suffix("2024Q1", "AAPL", "", False) != baseline
    assert suffix("2024Q1", "", "Tech", False) != baseline
    assert suffix("2024Q1", "", "", True) != baseline
    assert suffix("2024Q2", "", "", False) != baseline


def test_editor_key_is_stable_for_equivalent_filters(page_helpers) -> None:
    suffix = page_helpers["_editor_key_suffix"]
    assert suffix("2024Q1", " AAPL ", "Tech", True) == suffix("2024Q1", "aapl", "Tech", True)


# --- Metric coverage -------------------------------------------------------------------


def test_metric_coverage_counts_gaps(page_helpers, universe) -> None:
    coverage = page_helpers["_metric_coverage"](universe, ["ROE", "Debt"])

    by_metric = coverage.set_index("Metric")
    assert by_metric.loc["ROE", "Missing"] == 1
    assert by_metric.loc["ROE", "Present"] == 3
    assert by_metric.loc["ROE", "% Missing"] == pytest.approx(25.0)


def test_metric_coverage_skips_metrics_absent_from_the_table(page_helpers, universe) -> None:
    coverage = page_helpers["_metric_coverage"](universe, ["ROE", "NeverUploaded"])
    assert coverage["Metric"].tolist() == ["ROE"]


def test_metric_coverage_of_an_empty_table_does_not_divide_by_zero(page_helpers) -> None:
    empty = pd.DataFrame({"ticker": [], "ROE": []})
    coverage = page_helpers["_metric_coverage"](empty, ["ROE"])
    assert coverage.loc[0, "% Missing"] == 0.0
