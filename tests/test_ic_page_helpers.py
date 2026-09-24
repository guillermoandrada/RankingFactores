"""Table shaping and export for the Metric Selection (IC) page."""

from __future__ import annotations

import io
import pathlib

import pandas as pd
import pytest

_PAGE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "streamlit_app"
    / "pages"
    / "6_Metric_Selection.py"
)


@pytest.fixture(scope="module")
def page_helpers() -> dict:
    """
    Expose the page's module-level helpers without running its Streamlit body.

    Everything above the first `render_page_header` call is definitions only.
    """
    source = _PAGE.read_text(encoding="utf-8-sig")
    definitions = source.split("render_page_header(")[0]
    namespace: dict = {}
    exec(compile(definitions, str(_PAGE), "exec"), namespace)
    return namespace


@pytest.fixture
def ic_result() -> dict:
    return {
        "predictive": [
            {
                "metric_name": "ROE",
                "mean_rank_ic": 0.031,
                "ic_std": 0.12,
                "information_ratio": 0.26,
                "n_periods": 40,
            },
            {
                "metric_name": "Debt",
                "mean_rank_ic": -0.014,
                "ic_std": 0.09,
                "information_ratio": -0.15,
                "n_periods": 38,
            },
        ],
        "inter_factor_correlation": {
            "labels": ["ROE", "Debt"],
            "matrix": [[1.0, -0.42], [-0.42, 1.0]],
        },
        "periods": {
            "per_metric": [
                {
                    "metric_name": "ROE",
                    "available_periods": ["2024Q1", "2024Q2"],
                    "used_periods": ["2024Q1"],
                }
            ],
            "inter_factor": {"available_periods": ["2024Q1"]},
        },
        "warnings": [],
    }


# --- Table shaping ---------------------------------------------------------------------


def test_predictive_dataframe_columns(page_helpers, ic_result) -> None:
    frame = page_helpers["_predictive_dataframe"](ic_result["predictive"])

    assert list(frame.columns) == [
        "Metric",
        "Mean Rank IC",
        "IC Standard Deviation",
        "Information Ratio",
        "Periods (T)",
    ]
    assert frame.loc[0, "Metric"] == "ROE"
    assert frame.loc[1, "Periods (T)"] == 38


def test_predictive_dataframe_of_nothing_is_empty(page_helpers) -> None:
    assert page_helpers["_predictive_dataframe"]([]).empty


def test_correlation_dataframe_is_square_and_labelled(page_helpers, ic_result) -> None:
    frame = page_helpers["_correlation_dataframe"](ic_result["inter_factor_correlation"])

    assert list(frame.columns) == ["ROE", "Debt"]
    assert list(frame.index) == ["ROE", "Debt"]
    assert frame.loc["ROE", "Debt"] == pytest.approx(-0.42)


@pytest.mark.parametrize(
    "payload",
    [{}, {"labels": [], "matrix": []}, {"labels": ["ROE"], "matrix": []}],
)
def test_correlation_dataframe_handles_missing_matrix(page_helpers, payload) -> None:
    assert page_helpers["_correlation_dataframe"](payload).empty


def test_periods_dataframe_counts_available_and_used(page_helpers, ic_result) -> None:
    frame = page_helpers["_periods_dataframe"](ic_result["periods"])

    assert frame.loc[0, "Available periods"] == 2
    assert frame.loc[0, "Used periods"] == 1
    assert frame.loc[0, "Used period list"] == "2024Q1"


def test_periods_dataframe_of_nothing_is_empty(page_helpers) -> None:
    assert page_helpers["_periods_dataframe"]({}).empty


# --- Export ----------------------------------------------------------------------------


def test_export_writes_every_section(page_helpers, ic_result) -> None:
    content = page_helpers["_ic_export_bytes"].__wrapped__(ic_result)

    with pd.ExcelFile(io.BytesIO(content)) as workbook:
        assert workbook.sheet_names == ["Predictive", "Correlation", "Periods"]
        predictive = workbook.parse("Predictive")
        assert predictive["Metric"].tolist() == ["ROE", "Debt"]


def test_export_omits_sections_the_analysis_did_not_produce(page_helpers) -> None:
    sparse = {"predictive": [{"metric_name": "ROE", "n_periods": 1}]}

    content = page_helpers["_ic_export_bytes"].__wrapped__(sparse)

    with pd.ExcelFile(io.BytesIO(content)) as workbook:
        assert workbook.sheet_names == ["Predictive"]


def test_export_round_trips_the_correlation_matrix(page_helpers, ic_result) -> None:
    content = page_helpers["_ic_export_bytes"].__wrapped__(ic_result)

    with pd.ExcelFile(io.BytesIO(content)) as workbook:
        correlation = workbook.parse("Correlation", index_col=0)

    assert correlation.loc["ROE", "Debt"] == pytest.approx(-0.42)


# --- Group breakdown -------------------------------------------------------------------


_BREAKDOWN = {
    "group_by": "industry",
    "rows": [
        {"metric_name": "ROE", "group": "Banks", "mean_rank_ic": -0.05, "n_periods": 8, "mean_n": 20.0},
        {"metric_name": "ROE", "group": "Software", "mean_rank_ic": 0.07, "n_periods": 8, "mean_n": 30.0},
        {"metric_name": "Debt", "group": "Banks", "mean_rank_ic": 0.02, "n_periods": 7, "mean_n": 20.0},
    ],
}


def test_group_breakdown_dataframe_names_the_group_column(page_helpers) -> None:
    frame = page_helpers["_group_breakdown_dataframe"](_BREAKDOWN)

    assert list(frame.columns)[:2] == ["Metric", "Industry"]
    assert frame["Avg. securities"].tolist() == [20.0, 30.0, 20.0]


def test_group_ic_matrix_has_a_row_per_group_and_a_column_per_metric(page_helpers) -> None:
    matrix = page_helpers["_group_ic_matrix"](_BREAKDOWN)

    assert list(matrix.columns) == ["ROE", "Debt"]
    assert list(matrix.index) == ["Banks", "Software"]
    assert matrix.loc["Software", "ROE"] == pytest.approx(0.07)
    assert pd.isna(matrix.loc["Software", "Debt"])


@pytest.mark.parametrize("breakdown", [None, {"group_by": "sector", "rows": []}])
def test_group_breakdown_of_nothing_is_empty(page_helpers, breakdown) -> None:
    assert page_helpers["_group_breakdown_dataframe"](breakdown).empty
    assert page_helpers["_group_ic_matrix"](breakdown).empty


def test_export_adds_the_group_breakdown_sheet(page_helpers, ic_result) -> None:
    content = page_helpers["_ic_export_bytes"].__wrapped__({**ic_result, "group_breakdown": _BREAKDOWN})

    with pd.ExcelFile(io.BytesIO(content)) as workbook:
        assert workbook.sheet_names[-1] == "By industry"
        assert workbook.parse("By industry")["Industry"].tolist() == ["Banks", "Software", "Banks"]
