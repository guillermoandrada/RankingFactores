"""A derived metric must be analysable in IC exactly as a DB metric is."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from modules.domain.analytics.ic_analyzer import DerivedStep, ICAnalyzer, MetricPlan


class _StubStore:
    """Stands in for DerivedMetricStore with in-memory formulas."""

    def __init__(self, formulas: dict[str, dict]) -> None:
        self._formulas = formulas

    def list_formulas(self) -> dict[str, dict]:
        return self._formulas


class _StubAnalyzer(ICAnalyzer):
    """Drives the resolution and loading paths without a database."""

    def __init__(
        self,
        *,
        formulas: dict[str, dict],
        db_ids: dict[str, int],
        cross_sections: dict[int, pd.DataFrame] | None = None,
        periods: dict[int, list[str]] | None = None,
    ) -> None:
        self._derived_store = _StubStore(formulas)
        self._db_ids = db_ids
        self._cross_sections = cross_sections or {}
        self._periods = periods or {}
        self._logger = logging.getLogger("stub")

    def _db_metric_ids(self) -> dict[str, int]:
        return self._db_ids

    def _load_cross_section(self, *, metric_id: int, period: str) -> pd.DataFrame:
        return self._cross_sections.get(metric_id, pd.DataFrame())

    def _list_periods_for_metric(self, metric_id: int) -> list[str]:
        return self._periods.get(metric_id, [])


_FORMULAS = {
    "EBIT/EV": {
        "metric_names": ["EBIT", "EV"],
        "operations": ["/"],
        "higher_is_better": True,
    }
}
_DB_IDS = {"EBIT": 1, "EV": 2}


def _analyzer(**kwargs) -> _StubAnalyzer:
    return _StubAnalyzer(formulas=_FORMULAS, db_ids=_DB_IDS, **kwargs)


def test_a_derived_metric_resolves_to_its_base_metrics() -> None:
    warnings: list[str] = []

    plans = _analyzer()._resolve_metric_plans(["EBIT/EV"], warnings)

    plan = plans["EBIT/EV"]
    assert plan.base_ids == {"EBIT": 1, "EV": 2}
    assert plan.steps == (
        DerivedStep(name="EBIT/EV", metric_names=("EBIT", "EV"), operations=("/",)),
    )
    assert warnings == []


def test_a_db_metric_resolves_to_a_plan_without_steps() -> None:
    warnings: list[str] = []

    plans = _analyzer()._resolve_metric_plans(["EBIT"], warnings)

    assert plans["EBIT"] == MetricPlan(name="EBIT", base_ids={"EBIT": 1})
    assert not plans["EBIT"].is_derived


def test_an_unknown_metric_is_reported_rather_than_analysed() -> None:
    warnings: list[str] = []

    plans = _analyzer()._resolve_metric_plans(["Nonsense"], warnings)

    assert plans == {}
    assert len(warnings) == 1
    assert "Nonsense" in warnings[0]


def test_a_derived_cross_section_is_computed_from_its_base_metrics() -> None:
    analyzer = _analyzer(
        cross_sections={
            1: pd.DataFrame({"ticker": ["AAA", "BBB"], "value": [10.0, 20.0]}),
            2: pd.DataFrame({"ticker": ["AAA", "BBB"], "value": [100.0, 50.0]}),
        }
    )
    warnings: list[str] = []
    plan = analyzer._resolve_metric_plans(["EBIT/EV"], warnings)["EBIT/EV"]

    cross_section = analyzer._load_cross_section_for_plan(plan=plan, period="2025/03/30")

    assert cross_section["ticker"].tolist() == ["AAA", "BBB"]
    assert cross_section["value"].tolist() == [0.1, 0.4]


def test_tickers_missing_a_base_metric_drop_out_of_the_cross_section() -> None:
    """A formula has no value where one of its inputs is absent."""
    analyzer = _analyzer(
        cross_sections={
            1: pd.DataFrame({"ticker": ["AAA", "BBB"], "value": [10.0, 20.0]}),
            2: pd.DataFrame({"ticker": ["AAA"], "value": [100.0]}),
        }
    )
    warnings: list[str] = []
    plan = analyzer._resolve_metric_plans(["EBIT/EV"], warnings)["EBIT/EV"]

    cross_section = analyzer._load_cross_section_for_plan(plan=plan, period="2025/03/30")

    assert cross_section["ticker"].tolist() == ["AAA"]


@pytest.mark.parametrize(
    "periods, expected",
    [
        ({1: ["2025/03/30", "2025/06/30"], 2: ["2025/06/30"]}, ["2025/06/30"]),
        ({1: ["2025/03/30"], 2: []}, []),
    ],
)
def test_derived_availability_is_the_intersection_of_its_dependencies(periods, expected) -> None:
    analyzer = _analyzer(periods=periods)
    warnings: list[str] = []
    plan = analyzer._resolve_metric_plans(["EBIT/EV"], warnings)["EBIT/EV"]

    assert analyzer._list_periods_for_plan(plan) == expected
