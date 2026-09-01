"""An IC period that produces no point must say why it was skipped."""

from __future__ import annotations

import pandas as pd
import pytest

from modules.domain.analytics.ic_analyzer import ICAnalyzer, _summarise_periods


class _StubAnalyzer(ICAnalyzer):
    """Drives _ic_series_for_metric with controllable fundamentals and returns."""

    def __init__(self, *, fundamentals: pd.DataFrame, returns: pd.DataFrame) -> None:
        self._fundamentals = fundamentals
        self._returns = returns
        import logging

        self._logger = logging.getLogger("stub")

    def _parse_period_to_dates(self, period: str, forward_months: int):
        if period == "bad-label":
            raise ValueError("unparseable")
        return ("2025-05-14", "2025-08-14")

    def _load_cross_section(self, *, metric_id: int, period: str) -> pd.DataFrame:
        return self._fundamentals

    def _compute_forward_returns(self, *, tickers, start_date, end_date) -> pd.DataFrame:
        return self._returns


def _series(analyzer: _StubAnalyzer, periods=("2025/03/30",)):
    return analyzer._ic_series_for_metric(
        metric_id=1, forward_months=3, periods=list(periods)
    )


_VALUES = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"], "value": [1.0, 2.0, 3.0]})
_RETURNS = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"], "forward_return": [0.1, 0.2, 0.3]})


def test_a_usable_period_produces_a_point_and_no_warning() -> None:
    points, warnings = _series(_StubAnalyzer(fundamentals=_VALUES, returns=_RETURNS))

    assert len(points) == 1
    assert warnings == []


def test_missing_prices_are_explained_rather_than_silent() -> None:
    """This is what an empty Section A looked like before: no result, no reason."""
    analyzer = _StubAnalyzer(fundamentals=_VALUES, returns=pd.DataFrame())

    points, warnings = _series(analyzer)

    assert points == []
    assert len(warnings) == 1
    assert "no forward returns" in warnings[0]
    assert "2025-05-14 to 2025-08-14" in warnings[0]


def test_missing_fundamentals_are_explained() -> None:
    analyzer = _StubAnalyzer(fundamentals=pd.DataFrame(), returns=_RETURNS)

    _points, warnings = _series(analyzer)

    assert "no values in that period" in warnings[0]


def test_too_few_overlapping_securities_is_explained() -> None:
    analyzer = _StubAnalyzer(
        fundamentals=_VALUES,
        returns=pd.DataFrame({"ticker": ["AAA"], "forward_return": [0.1]}),
    )

    _points, warnings = _series(analyzer)

    assert "fewer than two securities" in warnings[0]


def test_an_unparseable_period_is_explained() -> None:
    analyzer = _StubAnalyzer(fundamentals=_VALUES, returns=_RETURNS)

    _points, warnings = _series(analyzer, periods=["bad-label"])

    assert "could not be parsed" in warnings[0]


def test_periods_failing_the_same_way_are_grouped_into_one_warning() -> None:
    """One warning per period would bury the explanation on a 40-period run."""
    analyzer = _StubAnalyzer(fundamentals=_VALUES, returns=pd.DataFrame())

    _points, warnings = _series(analyzer, periods=[f"2025/0{i}/30" for i in range(1, 6)])

    assert len(warnings) == 1
    assert "Skipped 5 of 5 period(s)" in warnings[0]


def test_analysis_prefixes_warnings_with_the_metric_name(monkeypatch) -> None:
    """A multi-metric run must say which factor each warning belongs to."""
    analyzer = _StubAnalyzer(fundamentals=_VALUES, returns=pd.DataFrame())
    monkeypatch.setattr(
        ICAnalyzer, "_resolve_metric_ids", lambda self, names, warnings: {"ROE": 1, "Debt": 2}
    )
    monkeypatch.setattr(
        ICAnalyzer, "_inter_factor_spearman_matrix", lambda self, **kwargs: [[1.0, 0.0], [0.0, 1.0]]
    )

    result = ICAnalyzer.analyze_multivariate(
        analyzer, metric_names=["ROE", "Debt"], forward_months=3, periods=["2025/03/30"]
    )

    assert any(w.startswith("ROE: ") for w in result["warnings"])
    assert any(w.startswith("Debt: ") for w in result["warnings"])


@pytest.mark.parametrize(
    "count, expected_tail",
    [(3, "c"), (8, "(+2 more)")],
)
def test_period_summary_truncates_long_lists(count, expected_tail) -> None:
    periods = [chr(ord("a") + i) for i in range(count)]

    assert _summarise_periods(periods).endswith(expected_tail)
