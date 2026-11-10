"""Rank IC can be broken down by sector or industry, next to the overall IC."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from modules.domain.analytics.ic_analyzer import ICAnalyzer, MetricPlan

_TICKERS = [f"T{i}" for i in range(12)]
# Tech: value and return move together; Banks: they move against each other.
_GROUPS = pd.DataFrame({"ticker": _TICKERS, "group": ["Tech"] * 6 + ["Banks"] * 6})
_VALUES = pd.DataFrame({"ticker": _TICKERS, "value": [float(i) for i in range(12)]})
_RETURNS = pd.DataFrame({
    "ticker": _TICKERS,
    "forward_return": [0.01 * i for i in range(6)] + [-0.01 * i for i in range(6)],
})


class _StubAnalyzer(ICAnalyzer):
    """Drives the grouped IC path with controllable fundamentals, returns, and groups."""

    def __init__(self, *, groups: pd.DataFrame = _GROUPS) -> None:
        self._groups = groups
        self.group_by_requests: list[str] = []
        self._logger = logging.getLogger("stub")

    def _resolve_metric_plans(self, names, warnings):
        return {name: MetricPlan(name=name, base_ids={name: i}) for i, name in enumerate(names)}

    def _inter_factor_spearman_matrix(self, **kwargs):
        return [[1.0, 0.0], [0.0, 1.0]]

    def _parse_period_to_dates(self, period: str, forward_months: int):
        return ("2025-05-14", "2025-08-14")

    def _load_cross_section(self, *, metric_id: int, period: str) -> pd.DataFrame:
        return _VALUES.copy()

    def _compute_forward_returns(self, *, tickers, start_date, end_date) -> pd.DataFrame:
        return _RETURNS

    def _load_security_groups(self, *, period: str, group_by: str) -> pd.DataFrame:
        self.group_by_requests.append(group_by)
        return self._groups


def _analyze(analyzer: _StubAnalyzer, group_by: str | None) -> dict:
    return analyzer.analyze_multivariate(
        metric_names=["ROE", "Debt"],
        forward_months=3,
        periods=["2025/03/30", "2025/06/30"],
        group_by=group_by,
    )


def test_without_group_by_there_is_no_breakdown() -> None:
    analyzer = _StubAnalyzer()

    result = _analyze(analyzer, None)

    assert result["group_breakdown"] is None
    assert analyzer.group_by_requests == []


def test_group_ic_is_computed_within_each_group() -> None:
    result = _analyze(_StubAnalyzer(), "industry")

    breakdown = result["group_breakdown"]
    assert breakdown["group_by"] == "industry"
    rows = {(r["metric_name"], r["group"]): r for r in breakdown["rows"]}
    assert rows[("ROE", "Tech")]["mean_rank_ic"] == pytest.approx(1.0)
    assert rows[("ROE", "Banks")]["mean_rank_ic"] == pytest.approx(-1.0)
    assert rows[("ROE", "Tech")]["n_periods"] == 2
    assert rows[("ROE", "Tech")]["mean_n"] == pytest.approx(6.0)


def test_the_overall_ic_is_unchanged_by_the_breakdown() -> None:
    overall = _analyze(_StubAnalyzer(), None)["predictive"]
    grouped = _analyze(_StubAnalyzer(), "sector")["predictive"]

    assert [p["mean_rank_ic"] for p in grouped] == [p["mean_rank_ic"] for p in overall]


def test_groups_below_the_minimum_size_are_left_out() -> None:
    groups = pd.DataFrame({"ticker": _TICKERS, "group": ["Tech"] * 8 + ["Tiny"] * 4})

    result = _analyze(_StubAnalyzer(groups=groups), "industry")

    assert {r["group"] for r in result["group_breakdown"]["rows"]} == {"Tech"}


def test_unclassified_securities_fall_into_unknown() -> None:
    groups = pd.DataFrame({"ticker": _TICKERS[:6], "group": ["Tech"] * 6})

    result = _analyze(_StubAnalyzer(groups=groups), "industry")

    assert {r["group"] for r in result["group_breakdown"]["rows"]} == {"Tech", "Unknown"}


def test_an_unknown_group_by_is_rejected() -> None:
    with pytest.raises(ValueError, match="group_by"):
        _analyze(_StubAnalyzer(), "country")
