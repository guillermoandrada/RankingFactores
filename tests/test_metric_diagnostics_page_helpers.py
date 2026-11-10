"""Metric Diagnostics page: DB and derived metrics are analysable alike."""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

_PAGE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "streamlit_app"
    / "pages"
    / "8_Metric_Diagnostics.py"
)

_FORMULAS = {
    "ROE Trend": {
        "metric_names": ["Estimated ROE", "Current ROE"],
        "operations": ["-"],
        "higher_is_better": True,
    }
}


@pytest.fixture
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


class _FakeStore:
    def list_formulas(self) -> dict:
        return _FORMULAS


class _FakeDb:
    engine = "fake-engine"

    def list_metrics(self) -> list[dict]:
        return [
            {"metric_id": 1, "metric_name": "Current ROE"},
            {"metric_id": 2, "metric_name": "Estimated ROE"},
        ]

    def list_periods(self) -> list[str]:
        return ["2024/03/31", "2024/06/30", "2024/09/30"]


def _matrix(values: list[float | None], column: str, *, missing: bool = False) -> pd.DataFrame:
    tickers = [f"T{i}" for i in range(len(values))]
    frame = pd.DataFrame(
        {column: values},
        index=pd.MultiIndex.from_arrays(
            [range(len(values)), tickers, tickers],
            names=["security_id", "ticker", "long_name"],
        ),
    )
    frame.attrs["missing_metrics"] = [column] if missing else []
    return frame


def _fake_fetch(calls: list[dict]):
    def fetch(engine, period, metric_names, derived_store, **kwargs):
        calls.append({"period": period, **kwargs})
        if period == "2024/09/30":
            return _matrix([None, None], metric_names[0], missing=True), {}
        return _matrix([0.1, None, 0.3], metric_names[0]), {}

    return fetch


def _fake_universe(requests: list[list[int]]):
    """2024/06/30 has no base rows; security 3 has a row whose values are all NULL."""

    def load(db, *, metric_ids):
        requests.append(sorted(metric_ids))
        return pd.DataFrame({
            "period": ["2024/03/31"] * 4 + ["2024/09/30"] * 2,
            "security_id": [0, 1, 2, 3, 0, 1],
        })

    return load


def test_metric_index_lists_db_and_derived_metrics(page_helpers) -> None:
    index = page_helpers["_metric_index"](_FakeDb(), _FakeStore())

    assert sorted(index) == ["Current ROE", "Estimated ROE", "ROE Trend"]
    assert page_helpers["is_derived_metric"](index["ROE Trend"])
    assert not page_helpers["is_derived_metric"](index["Current ROE"])


def test_derived_label_shows_its_formula(page_helpers) -> None:
    index = page_helpers["_metric_index"](_FakeDb(), _FakeStore())

    label = page_helpers["metric_label"](index, "ROE Trend")

    assert label == "ROE Trend  ·  derived: Estimated ROE - Current ROE"
    assert page_helpers["metric_label"](index, "Current ROE") == "Current ROE"


def test_derived_rows_are_raw_and_skip_unusable_periods(page_helpers) -> None:
    calls: list[dict] = []
    universe_requests: list[list[int]] = []
    page_helpers["fetch_metric_matrix"] = _fake_fetch(calls)
    page_helpers["_load_metric_universe"] = _fake_universe(universe_requests)
    index = page_helpers["_metric_index"](_FakeDb(), _FakeStore())

    rows = page_helpers["_load_metric_rows"](
        _FakeDb(), metric=index["ROE Trend"], derived_store=_FakeStore()
    )

    assert universe_requests == [[1, 2]]
    assert all(call["apply_na_handling"] is False for call in calls)
    assert [call["period"] for call in calls] == ["2024/03/31", "2024/09/30"]
    assert rows["period"].unique().tolist() == ["2024/03/31"]


def test_derived_rows_count_securities_missing_from_the_matrix_as_na(page_helpers) -> None:
    page_helpers["fetch_metric_matrix"] = _fake_fetch([])
    page_helpers["_load_metric_universe"] = _fake_universe([])
    index = page_helpers["_metric_index"](_FakeDb(), _FakeStore())

    rows = page_helpers["_load_metric_rows"](
        _FakeDb(), metric=index["ROE Trend"], derived_store=_FakeStore()
    )

    assert rows["security_id"].tolist() == [0, 1, 2, 3]
    assert rows["value"].isna().tolist() == [False, True, False, True]


def test_derived_formula_errors_are_raised_not_hidden(page_helpers) -> None:
    class _BrokenStore:
        def list_formulas(self) -> dict:
            return {"Broken": {"metric_names": ["Unknown Metric"], "operations": []}}

    page_helpers["_load_metric_universe"] = _fake_universe([])
    metric = {"metric_name": "Broken", "derived": True}

    with pytest.raises(ValueError, match="not found"):
        page_helpers["_load_metric_rows"](_FakeDb(), metric=metric, derived_store=_BrokenStore())


def test_diagnostics_count_missing_values_as_na(page_helpers) -> None:
    rows = pd.DataFrame({
        "period": ["2024/03/31"] * 4,
        "security_id": [1, 2, 3, 4],
        "value": [1.0, 2.0, 3.0, None],
    })

    payload = page_helpers["_compute_diagnostics"](rows, groups=None, group_label="")

    stats = payload["stats"].iloc[0]
    assert (stats["total"], stats["na_count"], stats["n"]) == (4, 1, 3)
    assert stats["na_pct"] == pytest.approx(25.0)
    assert payload["overall_mean"] == pytest.approx(2.0)
    assert payload["group_stats"] is None


def test_diagnostics_break_down_by_group_with_unknown_fallback(page_helpers) -> None:
    rows = pd.DataFrame({
        "period": ["2024/03/31"] * 3,
        "security_id": [1, 2, 3],
        "value": [1.0, 2.0, 3.0],
    })
    groups = pd.DataFrame({
        "period": ["2024/03/31", "2024/03/31"],
        "security_id": [1, 2],
        "group": ["Tech", "Tech"],
    })

    payload = page_helpers["_compute_diagnostics"](rows, groups=groups, group_label="Sector")

    by_group = payload["group_stats"].set_index("group")["n"].to_dict()
    assert by_group == {"Tech": 2, "Unknown": 1}
    assert payload["group_label"] == "Sector"
