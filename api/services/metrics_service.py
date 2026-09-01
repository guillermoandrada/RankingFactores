"""Metrics service: derived metric listing, validated create/update, and deletion."""

from __future__ import annotations

from typing import Any

import pandas as pd

from modules.domain.analytics.metric_loader import fetch_metric_matrix, validate_formula_graph

_PREVIEW_EXTREME_ROWS = 5


class _FormulaOverlay:
    """
    The stored formulas plus one unsaved candidate.

    fetch_metric_matrix only reads `list_formulas()`, so overlaying the candidate lets a
    preview reference existing derived metrics exactly as a saved formula would.
    """

    def __init__(
        self,
        store: Any,
        metric_name: str,
        metric_names: list[str],
        operations: list[str],
    ) -> None:
        self._store = store
        self._candidate = {
            metric_name: {
                "metric_names": metric_names,
                "operations": operations,
                "higher_is_better": None,
                # Left unset on purpose: filling gaps would hide the real gap rate.
                "na_handling": None,
            }
        }

    def list_formulas(self) -> dict[str, dict[str, Any]]:
        return {**self._store.list_formulas(), **self._candidate}


class DuplicateMetricError(ValueError):
    """A metric with that name already exists, as a derived formula or a DB metric."""


class MetricNotFoundError(ValueError):
    """The named derived metric does not exist."""


class MetricsService:
    """Owns derived metric rules: names must be unique and formulas must be computable."""

    def __init__(self, *, derived_store: Any, db: Any) -> None:
        self._derived_store = derived_store
        self._db = db

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

    def create_derived_metric(
        self,
        *,
        metric_name: str,
        metric_names: list[str],
        operations: list[str],
        higher_is_better: bool | None = None,
        na_handling: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a derived metric formula.

        Raises DuplicateMetricError when the name is taken, so a create can never
        overwrite an existing formula, and ValueError when the formula cannot be
        resolved to base metrics.
        """
        name = metric_name.strip()
        if not name:
            raise ValueError("metric_name cannot be empty.")

        if self._derived_store.get_formula(name) is not None:
            raise DuplicateMetricError(
                f"Derived metric '{name}' already exists. "
                "Edit it instead, or choose another name."
            )
        if name in self._base_metric_names():
            raise DuplicateMetricError(
                f"'{name}' is already a database metric. Derived metrics need a distinct name."
            )

        self._validate_formula(name, metric_names, operations)
        return self._derived_store.upsert_formula(
            metric_name=name,
            metric_names=metric_names,
            operations=operations,
            higher_is_better=higher_is_better,
            na_handling=na_handling,
        )

    def update_derived_metric(
        self,
        *,
        metric_name: str,
        metric_names: list[str] | None = None,
        operations: list[str] | None = None,
        higher_is_better: bool | None = None,
        na_handling: str | None = None,
    ) -> dict[str, Any]:
        """
        Update an existing derived metric formula.

        Raises MetricNotFoundError when it does not exist, and ValueError when the
        resulting formula would be unresolvable — including a cycle introduced by
        repointing this metric at one that depends on it.
        """
        current = self._derived_store.get_formula(metric_name)
        if current is None:
            raise MetricNotFoundError(f"Derived metric '{metric_name}' not found.")

        if metric_names is not None or operations is not None:
            self._validate_formula(
                metric_name,
                metric_names if metric_names is not None else current.get("metric_names", []),
                operations if operations is not None else current.get("operations", []),
            )

        self._derived_store.update_formula(
            metric_name=metric_name,
            metric_names=metric_names,
            operations=operations,
            higher_is_better=higher_is_better,
            na_handling=na_handling,
        )
        return {"metric_name": metric_name}

    def preview_derived_metric(
        self,
        *,
        period: str,
        metric_names: list[str],
        operations: list[str],
        metric_name: str | None = None,
        na_handling: str | None = None,
    ) -> dict[str, Any]:
        """
        Compute a candidate formula on one period without saving it.

        Reports the distribution and the extreme values, which is what catches a formula
        that is valid but wrong: a unit mismatch or an inverted sign shows up here rather
        than in a ranking weeks later.
        """
        name = (metric_name or "").strip() or "Preview metric"
        self._validate_formula(name, metric_names, operations)

        overlay = _FormulaOverlay(self._derived_store, name, metric_names, operations)
        matrix, _direction = fetch_metric_matrix(
            engine=self._db.engine,
            period=period,
            metric_names=[name],
            derived_store=overlay,
        )
        return self._summarise_column(matrix, name, period, na_handling)

    @staticmethod
    def _summarise_column(
        matrix: pd.DataFrame,
        metric_name: str,
        period: str,
        na_handling: str | None,
    ) -> dict[str, Any]:
        frame = matrix.reset_index()
        values = pd.to_numeric(frame.get(metric_name), errors="coerce")
        frame = frame.assign(_value=values)
        computed = frame.dropna(subset=["_value"])

        securities = int(len(frame))
        computed_count = int(len(computed))
        missing = securities - computed_count

        def extremes(subset: pd.DataFrame) -> list[dict[str, Any]]:
            ticker_column = "ticker" if "ticker" in subset.columns else None
            return [
                {
                    "ticker": str(row[ticker_column]) if ticker_column else "",
                    "value": float(row["_value"]),
                }
                for _, row in subset.iterrows()
            ]

        summary: dict[str, Any] = {
            "period": period,
            "metric_name": metric_name,
            "securities": securities,
            "computed": computed_count,
            "missing": missing,
            "missing_pct": (missing / securities * 100.0) if securities else 0.0,
            "na_handling": na_handling,
            "highest": [],
            "lowest": [],
        }
        for key in ("min", "p05", "median", "p95", "max", "mean", "std"):
            summary[key] = None

        if computed_count:
            series = computed["_value"]
            summary.update(
                {
                    "min": float(series.min()),
                    "p05": float(series.quantile(0.05)),
                    "median": float(series.median()),
                    "p95": float(series.quantile(0.95)),
                    "max": float(series.max()),
                    "mean": float(series.mean()),
                    "std": float(series.std(ddof=0)),
                    "highest": extremes(computed.nlargest(_PREVIEW_EXTREME_ROWS, "_value")),
                    "lowest": extremes(computed.nsmallest(_PREVIEW_EXTREME_ROWS, "_value")),
                }
            )
        return summary

    def delete_derived_metric(self, metric_name: str) -> None:
        """Delete a derived metric formula. Raises MetricNotFoundError when absent."""
        try:
            self._derived_store.delete_formula(metric_name)
        except ValueError as exc:
            raise MetricNotFoundError(str(exc)) from exc

    def _validate_formula(
        self,
        metric_name: str,
        metric_names: list[str],
        operations: list[str],
    ) -> None:
        """
        Check the formula resolves before it is stored.

        Without this the graph is only walked when a ranking runs, so a bad formula
        saves cleanly and fails much later, far from the change that caused it.
        """
        if len(metric_names) < 2:
            raise ValueError("metric_names must have at least 2 metrics.")
        if len(operations) != len(metric_names) - 1:
            raise ValueError(
                f"operations must have {len(metric_names) - 1} items "
                f"for {len(metric_names)} metrics."
            )

        prospective = dict(self._derived_store.list_formulas())
        prospective[metric_name] = {
            "metric_names": metric_names,
            "operations": operations,
        }
        validate_formula_graph(metric_name, prospective, self._base_metric_names())

    def _base_metric_names(self) -> set[str]:
        return {
            str(metric["metric_name"])
            for metric in self._db.list_metrics()
            if metric.get("metric_name")
        }
