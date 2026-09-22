"""IC analysis service with result caching."""

from __future__ import annotations

from functools import lru_cache

from modules.config.derived_metrics import DerivedMetricStore
from modules.domain.analytics.ic_analyzer import ICAnalyzer
from modules.infrastructure.db import FinancialDatabase
from modules.infrastructure.market_data.providers.base import BasePriceProvider


@lru_cache(maxsize=128)
def _cached_analyze_multivariate(
    analyzer: ICAnalyzer,
    metric_names: tuple[str, ...],
    forward_months: int,
    periods: tuple[str, ...] | None,
) -> dict:
    return analyzer.analyze_multivariate(
        metric_names=list(metric_names),
        forward_months=forward_months,
        periods=list(periods) if periods is not None else None,
    )


class ICService:
    def __init__(
        self,
        db: FinancialDatabase,
        price_provider: BasePriceProvider | None = None,
        derived_store: DerivedMetricStore | None = None,
    ) -> None:
        self._analyzer = ICAnalyzer(
            db=db, price_service=price_provider, derived_store=derived_store
        )

    def analyze(
        self,
        metric_names: list[str],
        forward_months: int,
        periods: list[str] | None = None,
    ) -> dict:
        """Run multivariate IC analysis. Results are cached per unique argument combination."""
        return _cached_analyze_multivariate(
            self._analyzer,
            tuple(metric_names),
            int(forward_months),
            tuple(periods) if periods is not None else None,
        )

    def invalidate_cache(self) -> None:
        """
        Clear every memoised IC input and result.

        Both the analysed results and the forward returns they were computed from, so a
        price upload or a new period cannot leave half of a run replaying old numbers.
        """
        _cached_analyze_multivariate.cache_clear()
        self._analyzer.clear_caches()
