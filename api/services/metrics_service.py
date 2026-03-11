"""Metrics service: derived metrics list and transformation."""

from __future__ import annotations

from typing import Any


class MetricsService:
    """Transforms derived metric store output for API consumption."""

    def __init__(self, *, derived_store: Any) -> None:
        self._derived_store = derived_store

    def list_derived_metrics(self) -> list[dict[str, Any]]:
        """
        Return all derived metric formulas in API-friendly shape.
        Each item: {metric_name, higher_is_better, na_handling, metric_names, operations}.
        """
        formulas = self._derived_store.list_formulas()
        return [
            {
                "metric_name": name,
                "higher_is_better": f.get("higher_is_better"),
                "na_handling": f.get("na_handling"),
                "metric_names": f.get("metric_names", []),
                "operations": f.get("operations", []),
            }
            for name, f in formulas.items()
        ]
