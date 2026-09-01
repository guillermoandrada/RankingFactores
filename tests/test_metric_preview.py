"""Computing a candidate derived metric on one period without saving it."""

from __future__ import annotations

import pandas as pd
import pytest

import api.services.metrics_service as metrics_service_module
from api.services.metrics_service import MetricsService


class _FakeStore:
    def __init__(self, formulas: dict | None = None) -> None:
        self.formulas = formulas or {}
        self.writes: list[str] = []

    def list_formulas(self) -> dict:
        return self.formulas

    def get_formula(self, metric_name: str):
        return self.formulas.get(metric_name)

    def upsert_formula(self, **kwargs):
        self.writes.append(kwargs["metric_name"])
        return kwargs


class _FakeDb:
    engine = "fake-engine"

    def __init__(self, names=("Debt", "Assets")) -> None:
        self._names = names

    def list_metrics(self) -> list[dict]:
        return [{"metric_name": name} for name in self._names]


def _matrix(values: list[float | None], column: str = "Candidate") -> pd.DataFrame:
    tickers = [f"T{i}" for i in range(len(values))]
    return pd.DataFrame(
        {column: values},
        index=pd.MultiIndex.from_arrays(
            [range(len(values)), tickers, tickers],
            names=["security_id", "ticker", "long_name"],
        ),
    )


@pytest.fixture
def service(monkeypatch):
    store = _FakeStore()
    built = MetricsService(derived_store=store, db=_FakeDb())
    captured: dict = {}

    def fake_fetch(*, engine, period, metric_names, derived_store, **kwargs):
        captured["engine"] = engine
        captured["period"] = period
        captured["metric_names"] = metric_names
        captured["formulas"] = derived_store.list_formulas()
        return _matrix(captured.get("values", [1.0, 2.0, 3.0]), metric_names[0]), {}

    monkeypatch.setattr(metrics_service_module, "fetch_metric_matrix", fake_fetch)
    return built, store, captured


def _preview(service, **overrides):
    built, _store, _captured = service
    payload = {
        "period": "2024/03/31",
        "metric_names": ["Debt", "Assets"],
        "operations": ["/"],
        "metric_name": "Candidate",
    }
    payload.update(overrides)
    return built.preview_derived_metric(**payload)


def test_preview_reports_the_distribution(service) -> None:
    _built, _store, captured = service
    captured["values"] = [1.0, 2.0, 3.0, 4.0]

    result = _preview(service)

    assert result["securities"] == 4
    assert result["computed"] == 4
    assert result["min"] == 1.0
    assert result["max"] == 4.0
    assert result["median"] == 2.5


def test_preview_counts_gaps(service) -> None:
    _built, _store, captured = service
    captured["values"] = [1.0, None, None, 4.0]

    result = _preview(service)

    assert result["computed"] == 2
    assert result["missing"] == 2
    assert result["missing_pct"] == pytest.approx(50.0)


def test_preview_returns_extremes_with_tickers(service) -> None:
    """The extremes are what reveal an inverted sign or a unit mismatch."""
    _built, _store, captured = service
    captured["values"] = [5.0, 1.0, 3.0]

    result = _preview(service)

    assert result["highest"][0]["value"] == 5.0
    assert result["highest"][0]["ticker"] == "T0"
    assert result["lowest"][0]["value"] == 1.0
    assert result["lowest"][0]["ticker"] == "T1"


def test_preview_handles_a_formula_that_computes_nothing(service) -> None:
    _built, _store, captured = service
    captured["values"] = [None, None]

    result = _preview(service)

    assert result["computed"] == 0
    assert result["min"] is None
    assert result["highest"] == []


def test_preview_overlays_the_candidate_on_stored_formulas(service) -> None:
    """The candidate must be visible to the resolver without being saved."""
    _built, store, captured = service
    store.formulas["Existing"] = {"metric_names": ["Debt", "Assets"], "operations": ["+"]}

    _preview(service)

    assert "Existing" in captured["formulas"]
    assert captured["formulas"]["Candidate"]["operations"] == ["/"]


def test_preview_does_not_apply_its_own_na_handling(service) -> None:
    """Filling gaps would hide the very gap rate the preview is reporting."""
    _preview(service, na_handling="replace_with_zero")
    _built, _store, captured = service

    assert captured["formulas"]["Candidate"]["na_handling"] is None


def test_preview_echoes_the_configured_na_handling(service) -> None:
    result = _preview(service, na_handling="replace_with_zero")

    assert result["na_handling"] == "replace_with_zero"


def test_preview_never_writes(service) -> None:
    _built, store, _captured = service

    _preview(service)

    assert store.writes == []


def test_preview_rejects_an_unknown_dependency(service) -> None:
    with pytest.raises(ValueError, match="NotAMetric"):
        _preview(service, metric_names=["Debt", "NotAMetric"])


def test_preview_rejects_a_mismatched_operation_count(service) -> None:
    with pytest.raises(ValueError, match="operations must have"):
        _preview(service, metric_names=["Debt", "Assets"], operations=["+", "-"])


def test_preview_falls_back_to_a_label_when_unnamed(service) -> None:
    result = _preview(service, metric_name=None)

    assert result["metric_name"] == "Preview metric"
