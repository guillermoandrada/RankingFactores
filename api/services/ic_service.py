"""IC analysis service with result caching."""

from __future__ import annotations

from functools import lru_cache

from modules.analytics.ic_analyzer import ICAnalyzer
from modules.db import FinancialDatabase


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
    def __init__(self, db: FinancialDatabase) -> None:
        self._analyzer = ICAnalyzer(db=db)

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
        """Clear the IC result cache (call after new period data is imported)."""
        _cached_analyze_multivariate.cache_clear()
