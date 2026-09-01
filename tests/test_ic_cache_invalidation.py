"""IC results are memoised, so anything that writes fundamentals must clear them."""

from __future__ import annotations

import pytest

import api.services.ic_service as ic_service_module
from api.services.ic_service import ICService


class _CountingAnalyzer:
    """Stands in for ICAnalyzer, counting how often a real computation happens."""

    def __init__(self) -> None:
        self.calls = 0

    def analyze_multivariate(self, *, metric_names, forward_months, periods):
        self.calls += 1
        return {"predictive": [], "calls": self.calls}


@pytest.fixture
def service_and_analyzer(monkeypatch):
    analyzer = _CountingAnalyzer()
    monkeypatch.setattr(ic_service_module, "ICAnalyzer", lambda **kwargs: analyzer)
    ic_service_module._cached_analyze_multivariate.cache_clear()
    yield ICService(db=object(), price_provider=None), analyzer
    ic_service_module._cached_analyze_multivariate.cache_clear()


def _run(service: ICService) -> dict:
    return service.analyze(metric_names=["ROE", "Debt"], forward_months=3)


def test_identical_requests_are_served_from_cache(service_and_analyzer) -> None:
    service, analyzer = service_and_analyzer
    _run(service)
    _run(service)

    assert analyzer.calls == 1


def test_a_different_request_is_computed(service_and_analyzer) -> None:
    service, analyzer = service_and_analyzer
    _run(service)
    service.analyze(metric_names=["ROE", "Debt"], forward_months=6)

    assert analyzer.calls == 2


def test_invalidate_cache_forces_a_recompute(service_and_analyzer) -> None:
    """
    Without this, importing a period leaves the next identical IC request replaying
    pre-import numbers while looking freshly computed.
    """
    service, analyzer = service_and_analyzer
    _run(service)
    service.invalidate_cache()
    _run(service)

    assert analyzer.calls == 2


def test_importing_fundamentals_invalidates_ic_results(monkeypatch) -> None:
    """The dependency hook is what the mutating period endpoints call."""
    from api import dependencies

    cleared: list[bool] = []

    class _Spy:
        def invalidate_cache(self) -> None:
            cleared.append(True)

    monkeypatch.setattr(dependencies, "get_ic_service", lambda: _Spy())
    dependencies.invalidate_fundamentals_caches()

    assert cleared == [True]


@pytest.mark.parametrize(
    "router_module, symbol",
    [
        ("api.routers.periods", "invalidate_fundamentals_caches"),
        ("api.routers.db_metrics", "invalidate_fundamentals_caches"),
    ],
)
def test_writers_of_fundamentals_import_the_hook(router_module, symbol) -> None:
    """A new write path that forgets to invalidate is the bug this guards against."""
    import importlib

    module = importlib.import_module(router_module)
    assert hasattr(module, symbol)
